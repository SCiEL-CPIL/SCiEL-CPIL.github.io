"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _compute_matrices():
    """
    Compute moment matrices for physicists' Hermite basis [H0, H4, H8, H12].
    M_ij = ∫ H_i H_j e^{-x^2} dx = sqrt(pi) * 2^n * n! * delta_ij
    A_ij = ∫ x^2 H_i H_j e^{-x^2} dx
    B_ij = ∫ H_i' H_j' e^{-x^2} dx
    """
    degrees = [0, 4, 8, 12]
    n = len(degrees)
    sp = np.sqrt(np.pi)
    
    M = np.zeros((n, n))
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    
    for i, deg_i in enumerate(degrees):
        M[i, i] = sp * (2 ** deg_i) * _factorial(deg_i)
        
        for j, deg_j in enumerate(degrees):
            val = 0.0
            if deg_i + 2 == deg_j:
                val += 0.25 * sp * (2 ** deg_j) * _factorial(deg_j)
            if deg_i == deg_j:
                val += deg_i * sp * (2 ** deg_i) * _factorial(deg_i)
            if deg_i >= 2 and deg_i - 2 == deg_j:
                val += (deg_i / 2.0) * sp * (2 ** deg_j) * _factorial(deg_j)
            A[i, j] = val
        
        for j, deg_j in enumerate(degrees):
            if deg_i > 0 and deg_j > 0 and deg_i == deg_j:
                B[i, j] = 4.0 * deg_i * deg_j * sp * (2 ** (deg_i - 1)) * _factorial(deg_i - 1)
    
    return M, A, B


def _bound_and_grad(params, M, A, B):
    """
    Compute bound and analytic gradient.
    params = [a, b] where c = [1, a, b, c12] with c12 = -(1+12a+1680b)/665280
    """
    a, b = params[0], params[1]
    c12 = -(1.0 + 12.0 * a + 1680.0 * b) / 665280.0
    c = np.array([1.0, a, b, c12])
    
    Mc = c @ M @ c
    Ac = c @ A @ c
    Bc = c @ B @ c
    
    if Mc <= 0:
        return 1e10, np.zeros(2)
    
    bound = (Ac * Bc) / (Mc * Mc)
    
    dc_da = np.array([0.0, 1.0, 0.0, -12.0 / 665280.0])
    dc_db = np.array([0.0, 0.0, 1.0, -1680.0 / 665280.0])
    
    dMc_da = 2.0 * c @ M @ dc_da
    dAc_da = 2.0 * c @ A @ dc_da
    dBc_da = 2.0 * c @ B @ dc_da
    
    dMc_db = 2.0 * c @ M @ dc_db
    dAc_db = 2.0 * c @ A @ dc_db
    dBc_db = 2.0 * c @ B @ dc_db
    
    grad_a = (dAc_da * Bc * Mc - Ac * dBc_da * Mc - 2.0 * Ac * Bc * dMc_da) / (Mc ** 3)
    grad_b = (dAc_db * Bc * Mc - Ac * dBc_db * Mc - 2.0 * Ac * Bc * dMc_db) / (Mc ** 3)
    
    return bound, np.array([grad_a, grad_b])


def construct():
    start_time = time.time()
    deadline = start_time + 165.0
    
    try:
        M, A, B = _compute_matrices()
        
        def obj(x):
            val, _ = _bound_and_grad(x, M, A, B)
            return val
        
        def obj_grad(x):
            _, g = _bound_and_grad(x, M, A, B)
            return g
        
        best_val = float('inf')
        best_params = None
        
        start_points = [
            [1.0/25.0, 1.0/3240.0],
            [0.0, 0.0],
            [0.04, 0.0003],
            [0.03, 0.0004],
            [0.05, 0.0002],
            [0.02, 0.0005],
            [0.06, 0.0001],
            [-0.04, 0.0003],
            [0.04, -0.0003],
            [0.01, 0.001],
            [0.1, 0.0001],
            [0.001, 0.01],
            [0.05, 0.0005],
            [0.02, 0.0002],
            [0.08, 0.0001],
        ]
        
        for sp in start_points:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(obj, sp, jac=obj_grad, method='L-BFGS-B',
                                  options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-14})
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x.copy()
            except Exception:
                pass
        
        nm_starts = [best_params.copy() if best_params is not None else np.array([0.04, 0.0003])]
        nm_starts.extend([[0.04, 0.0003], [0.03, 0.0004], [0.05, 0.0002]])
        
        for sp in nm_starts:
            if time.time() > deadline - 5:
                break
            try:
                result = minimize(obj, sp, method='Nelder-Mead',
                                  options={'maxiter': 50000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True})
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x.copy()
            except Exception:
                pass
        
        if best_params is not None and time.time() < deadline - 5:
            try:
                result = minimize(obj, best_params, method='Powell',
                                  options={'maxiter': 50000, 'xtol': 1e-15, 'ftol': 1e-16})
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x.copy()
            except Exception:
                pass
        
        if best_params is None:
            return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        
        a_opt, b_opt = best_params
        return np.array([1.0, a_opt, b_opt])
        
    except Exception:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
# EVOLVE-BLOCK-END
