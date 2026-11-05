import numpy as np
from scipy.linalg import expm
import time

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    n = 6
    dim = 2 ** n
    d = 0.8  # anisotropy
    h = 0.3  # field
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        ops = [ident] * n
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result
    
    def bond_op(i, j):
        sx_i = op_on_site(sx, i)
        sx_j = op_on_site(sx, j)
        sy_i = op_on_site(sy, i)
        sy_j = op_on_site(sy, j)
        sz_i = op_on_site(sz, i)
        sz_j = op_on_site(sz, j)
        return sx_i @ sx_j + sy_i @ sy_j + d * sz_i @ sz_j
    
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for i, j in odd_bonds:
        A += bond_op(i, j)
    for i, j in even_bonds:
        B += bond_op(i, j)
    
    half_field = 0.5 * h
    for site in range(n):
        A += half_field * op_on_site(sz, site)
        B += half_field * op_on_site(sz, site)
    
    return A, B

def _compute_error(a, b, A, B, times):
    dim = A.shape[0]
    H = A + B
    errors = []
    for t in times:
        U_exact = expm(-1j * t * H)
        U_approx = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            U_approx = expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A) @ U_approx
        err = np.linalg.norm(U_approx - U_exact, 2)
        errors.append(err)
    return np.mean(errors)

def coefficients(stages):
    t_start = time.time()
    deadline = t_start + 140.0
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    n = stages
    
    # Palindromic symmetry: 6 free a + 6 free b = 12 params
    half_n = n // 2
    
    def objective(params):
        a_local = np.zeros(n)
        b_local = np.zeros(n)
        for k in range(half_n):
            a_local[k] = params[k]
            a_local[n - 1 - k] = params[k]
            b_local[k] = params[half_n + k]
            b_local[n - 1 - k] = params[half_n + k]
        
        if abs(np.sum(a_local) - 1.0) > 1e-8 or abs(np.sum(b_local) - 1.0) > 1e-8:
            return 1e10
        if np.any(np.abs(a_local) > 50) or np.any(np.abs(b_local) > 50):
            return 1e10
        
        return _compute_error(a_local.tolist(), b_local.tolist(), A, B, times)
    
    # Initial: uniform
    x0 = np.full(half_n, 1.0 / n)
    x0 = np.concatenate([x0, x0])
    
    best_x = x0.copy()
    best_err = objective(x0)
    
    from scipy.optimize import minimize
    
    # Nelder-Mead with multiple restarts
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > deadline - 25.0:
            break
        if r == 0:
            x_cur = x0.copy()
        else:
            x_cur = x0 + np.random.randn(len(x0)) * 0.15
        try:
            res = minimize(objective, x_cur, method='Nelder-Mead',
                          options={'maxiter': 150, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Extract
    a_opt = np.zeros(n)
    b_opt = np.zeros(n)
    for k in range(half_n):
        a_opt[k] = best_x[k]
        a_opt[n - 1 - k] = best_x[k]
        b_opt[k] = best_x[half_n + k]
        b_opt[n - 1 - k] = best_x[half_n + k]
    
    # Normalize
    sa, sb = a_opt.sum(), b_opt.sum()
    if abs(sa) > 1e-12: a_opt /= sa
    if abs(sb) > 1e-12: b_opt /= sb
    
    a_opt = np.clip(a_opt, -50, 50)
    b_opt = np.clip(b_opt, -50, 50)
    sa, sb = a_opt.sum(), b_opt.sum()
    if abs(sa) > 1e-12: a_opt /= sa
    if abs(sb) > 1e-12: b_opt /= sb
    
    return a_opt.tolist(), b_opt.tolist()
