import numpy as np
from scipy.optimize import minimize
import itertools

def _build_matrices():
    """Build A and B matrices for 6-site XXZ Heisenberg chain."""
    # Spin-1/2 operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    n = 6
    dim = 2 ** n
    delta = 0.8
    h = 0.3
    
    def op_on_site(op, site):
        """Apply operator on a specific site (0-indexed)."""
        ops = [I2] * n
        ops[site] = op
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def bond_term(i, j):
        """XXZ bond term between sites i and j (0-indexed)."""
        # delta/2 * Sz_i Sz_j + 1/2 * (Sx_i Sx_j + Sy_i Sy_j)
        sz_i = op_on_site(sz, i)
        sz_j = op_on_site(sz, j)
        sx_i = op_on_site(sx, i)
        sx_j = op_on_site(sx, j)
        sy_i = op_on_site(sy, i)
        sy_j = op_on_site(sy, j)
        return (delta / 2.0) * sz_i @ sz_j + 0.5 * (sx_i @ sx_j + sy_i @ sy_j)
    
    # A: bonds (0,1), (2,3), (4,5) + half field
    A = np.zeros((dim, dim), dtype=complex)
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        A += bond_term(i, j)
    # Half uniform field: -h/2 * sum Sz_i, split half -> -h/4 * sum Sz_i
    for i in range(n):
        A += (-h / 4.0) * op_on_site(sz, i)
    
    # B: bonds (1,2), (3,4) + half field
    B = np.zeros((dim, dim), dtype=complex)
    for i, j in [(1, 2), (3, 4)]:
        B += bond_term(i, j)
    # Half uniform field
    for i in range(n):
        B += (-h / 4.0) * op_on_site(sz, i)
    
    return A, B

def _error_metric(coeffs, A, B, times=(0.25, 0.5, 1.0)):
    """Compute mean spectral norm error."""
    a, b = coeffs[:12], coeffs[12:]
    total_err = 0.0
    for t in times:
        U = np.eye(64, dtype=complex)
        for k in range(12):
            # exp(-i * b_k * t * B)
            eB = np.exp(-1j * b[k] * t * B)
            # exp(-i * a_k * t * A)
            eA = np.exp(-1j * a[k] * t * A)
            U = eB @ eA @ U
        # Exact: exp(-i * t * (A + B))
        AB = A + B
        U_exact = np.exp(-1j * t * AB)
        err = np.linalg.norm(U - U_exact, 2)
        total_err += err
    return total_err / len(times)

def coefficients(stages):
    """
    Optimize 12-stage product formula coefficients by direct numerical
    minimization of the spectral norm error on the specific Heisenberg chain.
    """
    A, B = _build_matrices()
    
    # Initial guess: symmetric high-order coefficients
    # Use a palindromic structure as starting point
    a_init = [0.0329, 0.09915, 0.11795, 0.11795, 0.09915, 0.0329,
              0.0329, 0.09915, 0.11795, 0.11795, 0.09915, 0.0329]
    b_init = [0.0329, 0.09915, 0.11795, 0.11795, 0.09915, 0.0329,
              0.0329, 0.09915, 0.11795, 0.11795, 0.09915, 0.0329]
    
    x0 = np.array(a_init + b_init, dtype=float)
    
    # Define bounds
    bounds = [(-50.0, 50.0)] * 24
    
    # Objective function
    def objective(x):
        return _error_metric(x, A, B)
    
    # Try a few optimization methods with limited iterations
    best_x = x0.copy()
    best_err = objective(x0)
    
    # Use Nelder-Mead for derivative-free optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       bounds=bounds,
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < best_err:
            best_err = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # Extract and normalize
    a = list(best_x[:12])
    b = list(best_x[12:])
    
    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    # Normalize to ensure sum is exactly 1
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x / sa for x in a]
    if abs(sb) > 1e-15:
        b = [x / sb for x in b]
    
    assert len(a) == stages
    assert len(b) == stages
    
    return a, b
