import numpy as np
from numpy.linalg import norm
import time
import sys

def coefficients(stages):
    """
    Computes optimal 12-stage product formula coefficients for a 6-site XXZ Heisenberg chain.
    Uses a global optimization approach (Nelder-Mead) on the coefficients, 
    minimizing the spectral norm error at t = 0.25, 0.5, 1.0.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Parameters from the task description
    N = 6
    anisotropy = 0.8
    field = 0.3
    
    # Construct the Hamiltonians A and B
    # The total Hamiltonian is H = A + B
    # A contains odd bonds (1-2, 3-4, 5-6) and half the field
    # B contains even bonds (2-3, 4-5) and half the field
    # The XXZ Heisenberg chain Hamiltonian is typically:
    # H = J * sum (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + dz Sz_i Sz_{i+1}) - h * sum Sz_i
    # Here J is the coupling constant. The task doesn't specify J, but usually J=1.
    # Let's assume J=1.
    # Sx = 0.5 * [[0, 1], [1, 0]]
    # Sy = 0.5 * [[0, -1j], [1j, 0]]
    # Sz = 0.5 * [[1, 0], [0, -1]]
    
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    I2 = np.eye(2, dtype=complex)
    
    def kron_chain(op_list):
        """Kronecker product of a list of operators."""
        result = np.array([[1]], dtype=complex)
        for op in op_list:
            result = np.kron(result, op)
        return result
    
    def op_on_site(i, op):
        """Return list of operators for kron, with op at site i and I2 elsewhere."""
        ops = [I2] * N
        ops[i] = op
        return ops
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)
    # Note: Python 0-indexed. Sites 1..6 in physics are 0..5 in Python.
    # Bond (1,2) -> indices (0,1) -> Odd
    # Bond (2,3) -> indices (1,2) -> Even
    # Bond (3,4) -> indices (2,3) -> Odd
    # Bond (4,5) -> indices (3,4) -> Even
    # Bond (5,6) -> indices (4,5) -> Odd
    
    # XXZ term: Sx_i Sx_j + Sy_i Sy_j + dz Sz_i Sz_j
    def xxz_term(i, j):
        return (np.kron_chain(op_on_site(i, sx)) @ np.kron_chain(op_on_site(j, sx)) +
                np.kron_chain(op_on_site(i, sy)) @ np.kron_chain(op_on_site(j, sy)) +
                anisotropy * np.kron_chain(op_on_site(i, sz)) @ np.kron_chain(op_on_site(j, sz)))
    
    # Field term: -h * sum Sz_i
    # Split field equally between A and B: -h/2 * sum Sz_i for each
    
    field_term = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        field_term += np.kron_chain(op_on_site(i, sz))
    field_term *= -field / 2.0
    
    # A: Odd bonds + half field
    A = xxz_term(0, 1) + xxz_term(2, 3) + xxz_term(4, 5) + field_term
    
    # B: Even bonds + half field
    B = xxz_term(1, 2) + xxz_term(3, 4) + field_term
    
    # Exact propagator: exp(-i t (A+B))
    # We need to compute this for t = 0.25, 0.5, 1.0
    # Using scipy.linalg.expm or eigen decomposition
    from scipy.linalg import expm
    
    H = A + B
    U_exact = {}
    for t in [0.25, 0.5, 1.0]:
        U_exact[t] = expm(-1j * t * H)
    
    def compute_error(a_list, b_list):
        """Compute the mean spectral norm error for given coefficients."""
        errors = []
        for t in [0.25, 0.5, 1.0]:
            U = np.eye(2**N, dtype=complex)
            # Product formula: prod_k exp(-i b_k t B) exp(-i a_k t A)
            # Order k=1..12
            for k in range(12):
                U = expm(-1j * b_list[k] * t * B) @ U
                U = expm(-1j * a_list[k] * t * A) @ U
            err = norm(U - U_exact[t], 2)
            errors.append(err)
        return np.mean(errors)
    
    # Initial guess: Suzuki 6th order coefficients
    c = 2.0 ** (1.0 / 3.0)
    denom = 1.0 - c
    a1 = 1.0 / (12.0 * denom)
    a2 = 1.0 / (24.0 * denom)
    a3 = 0.5 - 2.0 * a1 - 2.0 * a2
    b1 = 1.0 / (12.0 * denom)
    b2 = 1.0 / (24.0 * denom)
    b3 = 1.0 - 2.0 * b1 - 2.0 * b2
    
    # Map to 12 stages
    # The Suzuki scheme is symmetric.
    # We will use a 12-stage symmetric scheme directly.
    # Let's try to optimize 12 independent coefficients, but enforce symmetry to reduce search space?
    # Or just optimize all 24 coefficients (12 a's and 12 b's) with constraints sum(a)=1, sum(b)=1.
    # Nelder-Mead doesn't handle constraints well.
    # We can reparameterize: a_k = x_k / sum(x), b_k = y_k / sum(y).
    # But simpler: just optimize and project onto the constraint manifold?
    # Or use scipy.optimize.minimize with constraints.
    
    from scipy.optimize import minimize
    
    def objective(params):
        a_raw = params[:12]
        b_raw = params[12:]
        # Normalize to sum to 1
        sum_a = np.sum(a_raw)
        sum_b = np.sum(b_raw)
        if abs(sum_a) < 1e-10 or abs(sum_b) < 1e-10:
            return 1e10
        a_norm = a_raw / sum_a
        b_norm = b_raw / sum_b
        # Clip to [-50, 50]
        a_norm = np.clip(a_norm, -50, 50)
        b_norm = np.clip(b_norm, -50, 50)
        # Re-normalize after clipping? Might break sum=1.
        # If we clip, we must re-normalize.
        sum_a2 = np.sum(a_norm)
        sum_b2 = np.sum(b_norm)
        if abs(sum_a2) < 1e-10 or abs(sum_b2) < 1e-10:
            return 1e10
        a_final = a_norm / sum_a2
        b_final = b_norm / sum_b2
        
        # Check time budget
        if time.time() > deadline:
            return 1e10
            
        return compute_error(a_final, b_final)
    
    # Initial guess
    x0_a = np.array([a1, a2, a3, a2, a1, a1, a2, a3, a2, a1, 0.0, 0.0])
    x0_b = np.array([b1, b2, b3, b2, b1, b1, b2, b3, b2, b1, 0.0, 0.0])
    # Normalize x0
    x0_a = x0_a / np.sum(x0_a)
    x0_b = x0_b / np.sum(x0_b)
    x0 = np.concatenate([x0_a, x0_b])
    
    best_params = x0
    best_error = objective(x0)
    
    # Run Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < best_error:
            best_params = res.x
            best_error = res.fun
    except Exception as e:
        pass
    
    # Extract best coefficients
    a_best = best_params[:12]
    b_best = best_params[12:]
    a_best = a_best / np.sum(a_best)
    b_best = b_best / np.sum(b_best)
    a_best = np.clip(a_best, -50, 50)
    b_best = np.clip(b_best, -50, 50)
    # Final normalization
    a_best = a_best / np.sum(a_best)
    b_best = b_best / np.sum(b_best)
    
    return list(a_best), list(b_best)

