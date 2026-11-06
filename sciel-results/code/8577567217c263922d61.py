"""Optimized 12-stage product formula for Heisenberg chain time evolution."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

_START_TIME = time.time()
DEADLINE = 145.0  # safety margin before 150s kill

def _build_matrices():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    # 6-site open chain: sites 0-5
    # XXZ Hamiltonian: sum over bonds <ij> [Sx_i Sx_j + Sy_i Sy_j + anis * Sz_i Sz_j] + field * sum Sz_i
    # For spin-1/2: Sx = 0.5*sigma_x, etc.
    # Bond Hamiltonian for bond (i,j): 0.25*(sigma_x_i sigma_x_j + sigma_y_i sigma_y_j + anis*sigma_z_i sigma_z_j)
    # = 0.25*(sigma_+ sigma_- + sigma_- sigma_+) + 0.25*anis*sigma_z sigma_z
    # = 0.5*sigma_+ sigma_- + 0.25*anis*sigma_z sigma_z  (using symmetry)
    
    anis = 0.8
    field = 0.3
    n = 6
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    dim = 2 ** n
    
    def embed(op, site):
        """Embed a 2x2 operator at a given site in the n-site chain."""
        ops = [I2] * n
        ops[site] = op
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def bond_ham(i, j):
        """XXZ bond Hamiltonian between sites i and j."""
        # Sx_i Sx_j + Sy_i Sy_j + anis * Sz_i Sz_j
        # S = 0.5 * sigma
        # Sx_i Sx_j = 0.25 * sigma_x_i sigma_x_j
        # Sy_i Sy_j = 0.25 * sigma_y_i sigma_y_j
        # Sz_i Sz_j = 0.25 * sigma_z_i sigma_z_j
        term1 = 0.25 * embed(sx, i) @ embed(sx, j)
        term2 = 0.25 * embed(sy, i) @ embed(sy, j)
        term3 = 0.25 * anis * embed(sz, i) @ embed(sz, j)
        return term1 + term2 + term3
    
    def field_term(site, coeff):
        """Field term: coeff * Sz_site = coeff * 0.5 * sigma_z_site"""
        return coeff * 0.5 * embed(sz, site)
    
    # Bonds: (0,1), (1,2), (2,3), (3,4), (4,5)
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds go to A
    for i, j in [(0,1), (2,3), (4,5)]:
        A += bond_ham(i, j)
    
    # Even bonds go to B
    for i, j in [(1,2), (3,4)]:
        B += bond_ham(i, j)
    
    # Split field: half to A, half to B
    half_field = field / 2.0
    for site in range(n):
        A += field_term(site, half_field)
        B += field_term(site, half_field)
    
    return A, B

def _spectral_norm_error(coeffs, A, B, times):
    """Compute mean spectral norm error for given coefficients."""
    a = coeffs[:12]
    b = coeffs[12:24]
    
    total_err = 0.0
    for t in times:
        U = np.eye(64, dtype=complex)
        for k in range(12):
            # exp(-i b_k t B) exp(-i a_k t A)
            from scipy.linalg import expm
            U = np.linalg.multi_dot([expm(-1j * b[k] * t * B), expm(-1j * a[k] * t * A), U])
        
        exact = expm(-1j * t * (A + B))
        err = np.linalg.norm(U - exact, 2)  # spectral norm
        total_err += err
    
    return total_err / len(times)

def _optimize():
    """Optimize coefficients to minimize mean spectral norm error."""
    A, B = _build_matrices()
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: symmetric distribution
    a0 = [1.0/12] * 12
    b0 = [1.0/12] * 12
    x0 = np.array(a0 + b0)
    
    # Constraints: sum(a) = 1, sum(b) = 1, all in [-50, 50]
    # Linear constraints: sum(a[0:12]) = 1, sum(b[0:12]) = 1
    A_con = np.zeros((2, 24))
    A_con[0, :12] = 1.0
    A_con[1, 12:] = 1.0
    b_con = np.array([1.0, 1.0])
    
    bounds = [(-50.0, 50.0)] * 24
    
    constraint = LinearConstraint(A_con, b_con, b_con)
    
    def objective(x):
        return _spectral_norm_error(x, A, B, times)
    
    # Use L-BFGS-B with constraints
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                      constraints=[constraint], options={'maxiter': 100, 'ftol': 1e-10})
    
    if result.success:
        a_opt = result.x[:12]
        b_opt = result.x[12:]
        # Ensure exact sum = 1
        a_opt = a_opt / np.sum(a_opt)
        b_opt = b_opt / np.sum(b_opt)
        return list(a_opt), list(b_opt), _spectral_norm_error(result.x, A, B, times)
    else:
        # Fallback: use initial guess
        return list(a0), list(b0), _spectral_norm_error(x0, A, B, times)

def _try_local_refinement(a, b, A, B, times):
    """Try local refinement with different initial guesses."""
    best_a, best_b, best_err = a, b, _spectral_norm_error(np.array(a+b), A, B, times)
    
    # Try a few perturbations
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time.time() > DEADLINE - 10:
            break
        a_pert = np.array(a) + rng.normal(0, 0.1, 12)
        b_pert = np.array(b) + rng.normal(0, 0.1, 12)
        a_pert = np.clip(a_pert, -50, 50)
        b_pert = np.clip(b_pert, -50, 50)
        a_pert = a_pert / np.sum(a_pert)
        b_pert = b_pert / np.sum(b_pert)
        
        x_pert = np.concatenate([a_pert, b_pert])
        
        A_con = np.zeros((2, 24))
        A_con[0, :12] = 1.0
        A_con[1, 12:] = 1.0
        constraint = LinearConstraint(A_con, [1.0, 1.0], [1.0, 1.0])
        bounds = [(-50.0, 50.0)] * 24
        
        result = minimize(lambda x: _spectral_norm_error(x, A, B, times), x_pert,
                         method='SLSQP', bounds=bounds, constraints=[constraint],
                         options={'maxiter': 50, 'ftol': 1e-10})
        
        if result.success:
            a_new = result.x[:12] / np.sum(result.x[:12])
            b_new = result.x[12:] / np.sum(result.x[12:])
            err_new = _spectral_norm_error(result.x, A, B, times)
            if err_new < best_err:
                best_a, best_b, best_err = list(a_new), list(b_new), err_new
    
    return best_a, best_b, best_err

def coefficients(stages):
    """Find optimal 12-stage product formula coefficients."""
    if stages != 12:
        # Fallback for non-12 stages: uniform
        a = [1.0/stages] * stages
        b = [1.0/stages] * stages
        return a, b
    
    try:
        A, B = _build_matrices()
        times = [0.25, 0.5, 1.0]
        
        # First pass: SLSQP optimization
        a, b, err = _optimize()
        
        # Check time
        if time.time() > DEADLINE - 30:
            return a, b
        
        # Local refinement
        a, b, err = _try_local_refinement(a, b, A, B, times)
        
        # Final check
        if time.time() > DEADLINE - 5:
            pass
        
        return a, b
    
    except Exception:
        # Fallback: symmetric uniform
        a = [1.0/12] * 12
        b = [1.0/12] * 12
        return a, b
