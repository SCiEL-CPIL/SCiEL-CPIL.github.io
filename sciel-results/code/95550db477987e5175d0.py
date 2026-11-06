"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    anisotropy = 0.8
    field = 0.3
    
    # Full Hamiltonian: sum over bonds of (Sx*Sx + Sy*Sy + anisotropy*Sz*Sz) + field * sum Sz
    # For open chain, bonds are (0,1), (1,2), (2,3), (3,4), (4,5)
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    # Even bonds: (1,2), (3,4) -> B
    
    dim = 2 ** N
    I = np.eye(dim, dtype=complex)
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def kron_all(mats):
        """Tensor product of matrices in order (left to right = site 0 to N-1)."""
        result = np.array([[1]], dtype=complex)
        for m in mats:
            result = np.kron(result, m)
        return result
    
    def bond_term(i, j, mat1, mat2):
        """Compute operator mat1 on site i, mat2 on site j, identity elsewhere."""
        mats = [I2] * N
        mats[i] = mat1
        mats[j] = mat2
        return kron_all(mats)
    
    I2 = np.eye(2, dtype=complex)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Bonds: (0,1) odd, (1,2) even, (2,3) odd, (3,4) even, (4,5) odd
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        A += bond_term(i, j, sx, sx) + bond_term(i, j, sy, sy) + anisotropy * bond_term(i, j, sz, sz)
    
    for i, j in [(1, 2), (3, 4)]:
        B += bond_term(i, j, sx, sx) + bond_term(i, j, sy, sy) + anisotropy * bond_term(i, j, sz, sz)
    
    # Uniform field: field * sum_k Sz_k, split half into A, half into B
    field_half = field / 2.0
    for k in range(N):
        Sz_k = bond_term(k, k, sz, I2)  # This is wrong, need to fix
        # Actually, let's compute Sz on site k properly
        mats = [I2] * N
        mats[k] = sz
        Sz_k = kron_all(mats)
        A += field_half * Sz_k
        B += field_half * Sz_k
    
    return A, B

def _spectral_norm_error(a, b, A, B, times):
    """Compute mean spectral norm error ||U(t) - exp(-it(A+B))|| over given times."""
    dim = A.shape[0]
    total_error = 0.0
    n_times = len(times)
    
    for t in times:
        # Build U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            U = np.linalg.multi_dot([U, expm(-1j * b[k] * t * B), expm(-1j * a[k] * t * A)])
        
        # Exact propagator
        H = A + B
        U_exact = expm(-1j * t * H)
        
        # Spectral norm of difference
        diff = U - U_exact
        # Use singular values for spectral norm
        svals = np.linalg.svd(diff, compute_uv=False)
        total_error += svals[0]
    
    return total_error / n_times

def coefficients(stages):
    """Optimize 12-stage product formula coefficients to minimize mean spectral norm error."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: Strang splitting
    n = stages
    a0 = [0.5 / (n // 2)] * n
    b0 = []
    for _ in range(n // 2):
        b0 += [1.0 / (n // 2), 0.0]
    while len(b0) < n:
        b0.append(0.0)
    
    x0 = np.array(a0 + b0)
    
    def objective(x):
        a = x[:n]
        b = x[n:]
        # Enforce constraints via penalty
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        penalty = 1000.0 * (sum_a - 1.0) ** 2 + 1000.0 * (sum_b - 1.0) ** 2
        
        # Clip to valid range
        a_clip = np.clip(a, -50, 50)
        b_clip = np.clip(b, -50, 50)
        
        err = _spectral_norm_error(a_clip, b_clip, A, B, times)
        return err + penalty
    
    # Use L-BFGS-B with bounds
    bounds = [(-50, 50)] * (2 * n)
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Try a few optimization steps
    max_iter = 50
    for i in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        try:
            result = minimize(objective, best_x, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 5, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < best_val:
                best_x = result.x
                best_val = result.fun
        except Exception:
            break
    
    # Extract and normalize
    a = best_x[:n].copy()
    b = best_x[n:].copy()
    
    # Ensure sums are exactly 1
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if abs(sum_a - 1.0) > 1e-15:
        a = a / sum_a
    if abs(sum_b - 1.0) > 1e-15:
        b = b / sum_b
    
    # Final check
    a_list = [float(x) for x in a]
    b_list = [float(x) for x in b]
    
    return a_list, b_list
# EVOLVE-BLOCK-END
