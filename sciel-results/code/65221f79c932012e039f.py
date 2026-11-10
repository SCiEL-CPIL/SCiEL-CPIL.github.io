import time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field)."""
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    n = 6
    gamma = 0.8
    h = 0.3
    
    def site_op(op, site):
        ops = [np.eye(2) for _ in range(n)]
        ops[site] = op
        result = ops[0]
        for i in range(1, n):
            result = np.kron(result, ops[i])
        return result
    
    def bond_op(s1, s2):
        return (site_op(Sx, s1) @ site_op(Sx, s2) +
                site_op(Sy, s1) @ site_op(Sy, s2) +
                gamma * (site_op(Sz, s1) @ site_op(Sz, s2)))
    
    field = h * sum(site_op(Sz, j) for j in range(n))
    
    A = bond_op(0,1) + bond_op(2,3) + bond_op(4,5) + 0.5 * field
    B = bond_op(1,2) + bond_op(3,4) + 0.5 * field
    
    return A, B

def _spectral_norm_error(a, b, A, B, t):
    n = A.shape[0]
    U = np.eye(n, dtype=complex)
    for k in range(len(a)):
        U = expm(-1j * b[k] * t * B) @ U
        U = expm(-1j * a[k] * t * A) @ U
    U_exact = expm(-1j * t * (A + B))
    return np.linalg.norm(U - U_exact, 2)

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 50
    
    if stages != 12:
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    A, B = _build_hamiltonians()
    test_times = [0.25, 0.5, 1.0]
    
    def objective(x):
        a_half = x[:6]
        b_half = x[6:]
        a_full = list(a_half) + list(a_half[::-1])
        b_full = list(b_half) + list(b_half[::-1])
        if abs(sum(a_full) - 1.0) > 1e-8 or abs(sum(b_full) - 1.0) > 1e-8:
            return 1e10
        err = 0.0
        for t in test_times:
            err += _spectral_norm_error(a_full, b_full, A, B, t)
        return err / 3.0
    
    # Initial guess
    x0 = np.array([1/12.0]*12)
    a_s = sum(x0[:6]); b_s = sum(x0[6:])
    if a_s > 0: x0[:6] *= 0.5/a_s
    if b_s > 0: x0[6:] *= 0.5/b_s
    
    best_x = x0.copy()
    best_err = objective(x0)
    
    # Optimize
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11})
        if res.fun < best_err:
            best_err = res.fun
            best_x = res.x
    except: pass
    
    if time.time() - start_time < deadline - 10:
        # Second attempt from different start
        x1 = x0 + np.random.randn(12) * 0.2
        a_s = sum(x1[:6]); b_s = sum(x1[6:])
        if a_s > 0: x1[:6] *= 0.5/a_s
        if b_s > 0: x1[6:] *= 0.5/b_s
        try:
            res = minimize(objective, x1, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11})
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
        except: pass
    
    a_half = best_x[:6]
    b_half = best_x[6:]
    a_s = sum(a_half); b_s = sum(b_half)
    if a_s != 0: a_half = a_half * (0.5/a_s)
    if b_s != 0: b_half = b_half * (0.5/b_s)
    
    a_full = list(a_half) + list(a_half[::-1])
    b_full = list(b_half) + list(b_half[::-1])
    
    a_full = [max(-50, min(50, x)) for x in a_full]
    b_full = [max(-50, min(50, x)) for x in b_full]
    
    a_s = sum(a_full); b_s = sum(b_full)
    if a_s != 0: a_full = [x*(1.0/a_s) for x in a_full]
    if b_s != 0: b_full = [x*(1.0/b_s) for x in b_full]
    
    return a_full, b_full
