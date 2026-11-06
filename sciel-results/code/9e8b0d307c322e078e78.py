import numpy as np
import time
from scipy.optimize import least_squares, minimize

def _build_matrices():
    N = 6
    anis = 0.8
    field = 0.3
    dim = 2 ** N
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(mats):
        result = mats[0]
        for m in mats[1:]:
            result = np.kron(result, m)
        return result
    
    def bond_term(i, j, anis):
        terms = []
        mats = [I2] * N
        mats[i] = sx
        mats[j] = sx
        terms.append(0.5 * kron_all(mats))
        mats = [I2] * N
        mats[i] = sy
        mats[j] = sy
        terms.append(0.5 * kron_all(mats))
        mats = [I2] * N
        mats[i] = sz
        mats[j] = sz
        terms.append(anis * kron_all(mats))
        return sum(terms)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    for i, j in [(0,1), (2,3), (4,5)]:
        A += bond_term(i, j, anis)
    for i, j in [(1,2), (3,4)]:
        B += bond_term(i, j, anis)
    for site in range(N):
        mats = [I2] * N
        mats[site] = sz
        field_term = -field * kron_all(mats)
        A += 0.5 * field_term
        B += 0.5 * field_term
    return A, B

def _compute_error(a, b, A, B, times):
    errors = []
    for t in times:
        U = np.eye(2**6, dtype=complex)
        for k in range(len(a)):
            U = np.linalg.multi_dot([np.exp(-1j * b[k] * t * B), np.exp(-1j * a[k] * t * A), U])
        exact = np.exp(-1j * t * (A + B))
        err = np.linalg.norm(U - exact, 2)
        errors.append(err)
    return np.mean(errors)

def _solve_order_conditions(stages):
    n_half = stages // 2
    def residuals(x):
        a_half = x[:n_half]
        b_half = x[n_half:]
        r1 = np.sum(a_half) - 0.5
        r2 = np.sum(b_half) - 0.5
        r3 = np.sum(a_half**3) - 1.0/6.0
        r4 = np.sum(b_half**3) - 1.0/6.0
        return [r1, r2, r3, r4]
    
    x0 = np.concatenate([np.full(n_half, 0.5/n_half), np.full(n_half, 0.5/n_half)])
    try:
        sol = least_squares(residuals, x0, method='lm')
        a_half = sol.x[:n_half]
        b_half = sol.x[n_half:]
        a = np.concatenate([a_half, a_half[::-1]])
        b = np.concatenate([b_half, b_half[::-1]])
        if np.all(np.abs(a) <= 50) and np.all(np.abs(b) <= 50):
            return a, b
    except Exception:
        pass
    return None

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140.0
    A, B = _build_matrices()
    times = [0.25, 0.5, 1.0]
    
    result = _solve_order_conditions(stages)
    if result is not None:
        a, b = result
        best_a, best_b = a, b
        best_err = _compute_error(a, b, A, B, times)
        
        # Refine with minimize if time allows
        if time.time() < deadline - 20.0:
            def objective(x):
                a_opt = np.concatenate([x[:6], x[:6][::-1]])
                b_opt = np.concatenate([x[6:12], x[6:12][::-1]])
                # Normalize to ensure sum=1? No, we solved for sum=1.
                # But small numerical errors might creep in.
                sa = np.sum(a_opt)
                sb = np.sum(b_opt)
                if sa != 0: a_opt = a_opt / sa
                if sb != 0: b_opt = b_opt / sb
                if np.any(np.abs(a_opt) > 50) or np.any(np.abs(b_opt) > 50):
                    return 1e10
                return _compute_error(a_opt, b_opt, A, B, times)
            
            x0 = np.concatenate([best_a[:6], best_b[:6]])
            try:
                res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_err:
                    a_opt = np.concatenate([res.x[:6], res.x[:6][::-1]])
                    b_opt = np.concatenate([res.x[6:12], res.x[6:12][::-1]])
                    sa = np.sum(a_opt)
                    sb = np.sum(b_opt)
                    if sa != 0: a_opt = a_opt / sa
                    if sb != 0: b_opt = b_opt / sb
                    if np.all(np.abs(a_opt) <= 50) and np.all(np.abs(b_opt) <= 50):
                        best_a, best_b = a_opt, b_opt
            except Exception:
                pass
            
        return list(best_a), list(best_b)
        
    a = [1.0/stages] * stages
    b = [1.0/stages] * stages
    return a, b
