import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite
import time

# Pre-compute Gauss-Hermite nodes and weights
_N_GH = 100
_t_nodes, _w_weights = np.polynomial.hermite.hermgauss(_N_GH)

_c2 = 1.0 / np.sqrt(2.0 * np.pi)
_c4 = 1.0 / np.sqrt(4.0 * np.pi)

_x2_nodes = _t_nodes * _c2
_x4_nodes = _t_nodes * _c4

_w2_nodes = _w_weights * _c2
_w4_nodes = _w_weights * _c4

_H0_2 = np.ones_like(_x2_nodes)
_H4_2 = eval_hermite(4, _x2_nodes)
_H8_2 = eval_hermite(8, _x2_nodes)
_H12_2 = eval_hermite(12, _x2_nodes)

_H0_4 = np.ones_like(_x4_nodes)
_H4_4 = eval_hermite(4, _x4_nodes)
_H8_4 = eval_hermite(8, _x4_nodes)
_H12_4 = eval_hermite(12, _x4_nodes)

def _compute_objective_fast(coeffs):
    a0, a4, a8 = coeffs
    a12 = (a0 - 12.0 * a4 + 1680.0 * a8) / 332640.0
    
    P2 = a0 * _H0_2 + a4 * _H4_2 + a8 * _H8_2 + a12 * _H12_2
    P4 = a0 * _H0_4 + a4 * _H4_4 + a8 * _H8_4 + a12 * _H12_4
    
    Pm2 = a0 * _H0_2 + a4 * _H4_2 + a8 * _H8_2 - a12 * _H12_2
    Pm4 = a0 * _H0_4 + a4 * _H4_4 + a8 * _H8_4 - a12 * _H12_4
    
    norm2_sq = np.sum(P2**2 * _w2_nodes)
    norm4_sq = np.sum(P4**4 * _w4_nodes)
    norm2_sq_hat = np.sum(Pm2**2 * _w2_nodes)
    norm4_sq_hat = np.sum(Pm4**4 * _w4_nodes)
    
    if norm2_sq < 1e-12 or norm2_sq_hat < 1e-12:
        return 1e10
        
    obj = (norm4_sq / (norm2_sq * norm2_sq)) * (norm4_sq_hat / (norm2_sq_hat * norm2_sq_hat))
    return obj

def construct():
    start_time = time.time()
    deadline = 180.0
    
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_result = x0.copy()
    best_score = _compute_objective_fast(x0)
    
    # Phase 1: Local optimization from seed and diverse perturbations
    np.random.seed(42)
    starts = [x0.copy()]
    for scale in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        for i in range(4):
            perturbation = np.random.randn(3) * np.array([scale, scale * 0.1, scale * 0.001])
            starts.append(x0 + perturbation)
    
    for s in starts:
        if time.time() > deadline - 90:
            break
        try:
            res = minimize(_compute_objective_fast, s, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Differential Evolution with expanded bounds and multiple runs
    bounds = [(-20.0, 20.0), (-5.0, 5.0), (-0.1, 0.1)]
    
    if time.time() < deadline - 50:
        try:
            de_res = differential_evolution(_compute_objective_fast, bounds,
                                            seed=42, maxiter=150, tol=1e-10,
                                            polish=True, init='sobol', popsize=20)
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_result = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Second DE run with different seed for robustness
    if time.time() < deadline - 30:
        try:
            de_res2 = differential_evolution(_compute_objective_fast, bounds,
                                             seed=123, maxiter=100, tol=1e-10,
                                             polish=True, init='latinhypercube', popsize=15)
            if de_res2.fun < best_score:
                best_score = de_res2.fun
                best_result = de_res2.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 15:
        try:
            res_nm = minimize(_compute_objective_fast, best_result, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True})
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_result = res_nm.x.copy()
        except Exception:
            pass
    
    # Phase 5: Ultra-tight L-BFGS-B polish
    if time.time() < deadline - 5:
        try:
            res_final = minimize(_compute_objective_fast, best_result, method='L-BFGS-B',
                                options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-18})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_result = res_final.x.copy()
        except Exception:
            pass
    
    return best_result
