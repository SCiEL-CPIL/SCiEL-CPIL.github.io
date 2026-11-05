"""Optimized twelve-stage product formula for Heisenberg-chain time evolution."""
import time
import numpy as np
from scipy.linalg import expm

# We will compute the optimal coefficients via optimization.
# The task: U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
# Score: mean over t in {0.25, 0.5, 1.0} of ||U(t) - exp(-i t (A+B))||_spectral

def _build_hamiltonians():
    """Build A and B for the 6-site open XXZ Heisenberg chain.
    
    XXZ Hamiltonian: H = sum_{j=1}^{N-1} [Jx (Sx_j Sx_{j+1} + Sy_j Sy_{j+1}) + Jz Sz_j Sz_{j+1}] + h sum_j Sz_j
    With anisotropy 0.8 (Jz/Jx = 0.8), field 0.3.
    
    Convention: Jx = 1, Jz = 0.8, h = 0.3
    N = 6 sites, spin-1/2.
    
    A = odd bonds (1-2, 3-4, 5-6) + half field
    B = even bonds (2-3, 4-5) + half field
    """
    N = 6
    Jx = 1.0
    Jz = 0.8
    h = 0.3
    
    # Spin-1/2 matrices
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Build full Hamiltonian terms
    # We'll use Kronecker products
    def op_on_site(op, site):
        """Apply op at 'site' (0-indexed) in N-site chain."""
        result = np.array([[1.0]])
        for s in range(N):
            if s == site:
                result = np.kron(result, op)
            else:
                result = np.kron(result, np.eye(2))
        return result
    
    def op_on_sites(op1, site1, op2, site2):
        """Apply op1 at site1 and op2 at site2."""
        result = np.array([[1.0]])
        for s in range(N):
            if s == site1:
                result = np.kron(result, op1)
            elif s == site2:
                result = np.kron(result, op2)
            else:
                result = np.kron(result, np.eye(2))
        return result
    
    # Odd bonds: (0,1), (2,3), (4,5)
    A = np.zeros((2**N, 2**N), dtype=complex)
    for j in [0, 2, 4]:
        A += Jx * (op_on_sites(Sx, j, Sx, j+1) + op_on_sites(Sy, j, Sy, j+1))
        A += Jz * op_on_sites(Sz, j, Sz, j+1)
    
    # Even bonds: (1,2), (3,4)
    B = np.zeros((2**N, 2**N), dtype=complex)
    for j in [1, 3]:
        B += Jx * (op_on_sites(Sx, j, Sx, j+1) + op_on_sites(Sy, j, Sy, j+1))
        B += Jz * op_on_sites(Sz, j, Sz, j+1)
    
    # Field: half on A, half on B
    field_term = np.zeros((2**N, 2**N), dtype=complex)
    for j in range(N):
        field_term += h * op_on_site(Sz, j)
    
    A += 0.5 * field_term
    B += 0.5 * field_term
    
    return A, B


def _compute_error(a, b, A, B, t_values):
    """Compute the spectral norm error for given coefficients."""
    total_err = 0.0
    for t in t_values:
        U = np.eye(2**6, dtype=complex)
        for k in range(len(a)):
            U = expm(-1j * b[k] * t * B) @ U
            U = expm(-1j * a[k] * t * A) @ U
        U_exact = expm(-1j * t * (A + B))
        # Spectral norm = largest singular value
        err = np.linalg.norm(U - U_exact, 2)
        total_err += err
    return total_err / len(t_values)


def _optimize_coefficients(A, B, t_values, deadline):
    """Optimize coefficients using Nelder-Mead or similar."""
    from scipy.optimize import minimize
    
    stages = 12
    n_params = 2 * stages  # a_1..a_12, b_1..b_12
    
    def objective(x):
        a = list(x[:stages])
        b = list(x[stages:])
        # Enforce sum constraints by projecting
        a_sum = sum(a)
        b_sum = sum(b)
        a = [ai + (1.0 - a_sum) / stages for ai in a]
        b = [bi + (1.0 - b_sum) / stages for bi in b]
        # Clamp to [-50, 50]
        a = [max(-50.0, min(50.0, ai)) for ai in a]
        b = [max(-50.0, min(50.0, bi)) for bi in b]
        return _compute_error(a, b, A, B, t_values)
    
    # Initial guess: equal distribution
    x0 = [1.0/stages] * (2*stages)
    
    best_x = x0.copy()
    best_f = objective(x0)
    
    start_time = time.time()
    
    # Try Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10,
                               'adaptive': True})
        if time.time() - start_time > deadline - 5:
            raise TimeoutError()
        a = list(res.x[:stages])
        b = list(res.x[stages:])
        a_sum = sum(a)
        b_sum = sum(b)
        a = [ai + (1.0 - a_sum) / stages for ai in a]
        b = [bi + (1.0 - b_sum) / stages for bi in b]
        a = [max(-50.0, min(50.0, ai)) for ai in a]
        b = [max(-50.0, min(50.0, bi)) for bi in b]
        f = objective(res.x)
        if f < best_f:
            best_f = f
            best_x = res.x.copy()
    except Exception:
        pass
    
    if time.time() - start_time > deadline - 3:
        a = list(best_x[:stages])
        b = list(best_x[stages:])
        a_sum = sum(a)
        b_sum = sum(b)
        a = [ai + (1.0 - a_sum) / stages for ai in a]
        b = [bi + (1.0 - b_sum) / stages for bi in b]
        a = [max(-50.0, min(50.0, ai)) for ai in a]
        b = [max(-50.0, min(50.0, bi)) for bi in b]
        return a, b
    
    # Try L-BFGS-B with penalty for constraints
    try:
        def obj_l(x):
            a = list(x[:stages])
            b = list(x[stages:])
            a_sum = sum(a)
            b_sum = sum(b)
            a = [ai + (1.0 - a_sum) / stages for ai in a]
            b = [bi + (1.0 - b_sum) / stages for bi in b]
            a = [max(-50.0, min(50.0, ai)) for ai in a]
            b = [max(-50.0, min(50.0, bi)) for bi in b]
            return _compute_error(a, b, A, B, t_values)
        
        bounds = [(-50.0, 50.0)] * n_params
        res2 = minimize(obj_l, best_x, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        if time.time() - start_time > deadline - 2:
            raise TimeoutError()
        a = list(res2.x[:stages])
        b = list(res2.x[stages:])
        a_sum = sum(a)
        b_sum = sum(b)
        a = [ai + (1.0 - a_sum) / stages for ai in a]
        b = [bi + (1.0 - b_sum) / stages for bi in b]
        a = [max(-50.0, min(50.0, ai)) for ai in a]
        b = [max(-50.0, min(50.0, bi)) for bi in b]
        f2 = obj_l(res2.x)
        if f2 < best_f:
            best_f = f2
            best_x = res2.x.copy()
    except Exception:
        pass
    
    a = list(best_x[:stages])
    b = list(best_x[stages:])
    a_sum = sum(a)
    b_sum = sum(b)
    a = [ai + (1.0 - a_sum) / stages for ai in a]
    b = [bi + (1.0 - b_sum) / stages for bi in b]
    a = [max(-50.0, min(50.0, ai)) for ai in a]
    b = [max(-50.0, min(50.0, bi)) for bi in b]
    return a, b


# Cache the result
_cache = None

def coefficients(stages):
    """Return optimized coefficients for the product formula."""
    global _cache
    if _cache is not None:
        return _cache
    
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    A, B = _build_hamiltonians()
    t_values = [0.25, 0.5, 1.0]
    
    if time.time() - start_time > deadline - 10:
        # Fallback: uniform
        a = [1.0/stages] * stages
        b = [1.0/stages] * stages
        _cache = (a, b)
        return a, b
    
    a, b = _optimize_coefficients(A, B, t_values, deadline - 5)
    _cache = (a, b)
    return a, b
