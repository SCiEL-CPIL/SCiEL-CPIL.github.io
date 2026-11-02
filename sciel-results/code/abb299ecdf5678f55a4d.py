# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial

# Precompute constants
_N_VALS = np.array([0, 4, 8, 12], dtype=np.float64)
_W_VALS = (2.0 ** _N_VALS) * factorial(_N_VALS, exact=False)
_V_VALS = _N_VALS + 0.5  # n + 0.5
_U_VALS = 2.0 * _N_VALS + 1.0  # 2n + 1

# H_n(0) for physicists' Hermite polynomials
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 665280.0

def _compute_bound_analytic(coeffs):
    """
    Compute the upper bound on C4 = Var(x) * Var(k).
    coeffs = [a, b, c] for H0, H4, H8; H12 coefficient derived from g(0)=0.
    """
    a, b, c = coeffs
    
    # Derive d from constraint g(0) = 0:
    # a*H0(0) + b*H4(0) + c*H8(0) + d*H12(0) = 0
    d = -(a * _H0_0 + b * _H4_0 + c * _H8_0) / _H12_0
    
    c_vals = np.array([a, b, c, d], dtype=np.float64)
    c_sq = c_vals * c_vals
    
    sum_w = np.dot(c_sq, _W_VALS)
    
    if sum_w < 1e-30:
        return 1e10
    
    sum_vw = np.dot(c_sq, _V_VALS * _W_VALS)
    sum_uw = np.dot(c_sq, _U_VALS * _W_VALS)
    
    var_x = sum_vw / sum_w
    var_k = sum_uw / sum_w
    
    return var_x * var_k

def _compute_bound_grad(coeffs):
    """
    Compute the bound and its gradient with respect to [a, b, c].
    """
    a, b, c = coeffs
    d = -(a * _H0_0 + b * _H4_0 + c * _H8_0) / _H12_0
    
    c_vals = np.array([a, b, c, d], dtype=np.float64)
    c_sq = c_vals * c_vals
    
    sum_w = np.dot(c_sq, _W_VALS)
    
    if sum_w < 1e-30:
        return 1e10, np.zeros(3)
    
    sum_vw = np.dot(c_sq, _V_VALS * _W_VALS)
    sum_uw = np.dot(c_sq, _U_VALS * _W_VALS)
    
    var_x = sum_vw / sum_w
    var_k = sum_uw / sum_w
    bound = var_x * var_k
    
    # Gradient computation
    # Let S = sum_w, Sv = sum_vw, Su = sum_uw
    # var_x = Sv/S, var_k = Su/S
    # d(bound)/d(c_n) = var_k * d(Sv/S)/d(c_n) + var_x * d(Su/S)/d(c_n)
    # d(Sv/S)/d(c_n) = (2*c_n*v_n*w_n*S - Sv*2*c_n*w_n) / S^2
    #                 = 2*c_n*w_n*(v_n*S - Sv) / S^2
    # Similarly for Su.
    
    # We need dc_vals/dc_coeffs for each of a, b, c
    # c_vals = [a, b, c, d] where d = -(a + 12b + 1680c)/665280
    # dd/da = -1/665280, dd/db = -12/665280, dd/dc = -1680/665280
    
    dd_da = -1.0 / _H12_0
    dd_db = -12.0 / _H12_0
    dd_dc = -1680.0 / _H12_0
    
    grad = np.zeros(3)
    
    for i, dc_coeff in enumerate([(1.0, 0.0, 0.0, dd_da), (0.0, 1.0, 0.0, dd_db), (0.0, 0.0, 1.0, dd_dc)]):
        # dc_vals/d(coeff_i)
        dc0, dc1, dc2, dc3 = dc_coeff
        # d(c_sq)/d(coeff_i) = 2 * c_vals * dc_vals
        dc_sq = 2.0 * c_vals * np.array([dc0, dc1, dc2, dc3])
        
        dS = np.dot(dc_sq, _W_VALS)
        dSv = np.dot(dc_sq, _V_VALS * _W_VALS)
        dSu = np.dot(dc_sq, _U_VALS * _W_VALS)
        
        # d(var_x)/d(coeff_i) = (dSv*S - Sv*dS) / S^2
        dvar_x = (dSv * sum_w - sum_vw * dS) / (sum_w * sum_w)
        dvar_k = (dSu * sum_w - sum_uw * dS) / (sum_w * sum_w)
        
        grad[i] = var_k * dvar_x + var_x * dvar_k
    
    return bound, grad

def construct():
    """Find optimal coefficients [a, b, c] minimizing the C4 upper bound."""
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    best_fun = 1e10
    best_x = np.array([1.0, 0.0, 0.0])
    
    def _update_best(fun, x):
        nonlocal best_fun, best_x
        if fun < best_fun:
            best_fun = fun
            best_x = np.array(x, dtype=np.float64)
    
    # Phase 1: Global search with differential evolution
    # The function is scale-invariant in a sense (multiplying all coeffs by same factor doesn't change bound),
    # but the constraint breaks this. Let's bound the search reasonably.
    # From analysis, optimal coefficients should be O(1).
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    
    if time.time() < deadline - 30.0:
        try:
            de_res = differential_evolution(
                _compute_bound_analytic,
                bounds=bounds,
                seed=42,
                maxiter=1000,
                tol=1e-12,
                popsize=20,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,  # We'll polish ourselves
                init='latinhypercube',
                update_interval=50,
                callback=None
            )
            if de_res.fun < best_fun:
                _update_best(de_res.fun, de_res.x)
        except Exception:
            pass
    
    # Phase 2: Local refinement with L-BFGS-B using analytic gradient
    # Start from the DE result and several other good points
    local_starts = [best_x.copy()]
    # Add some heuristic starting points
    heuristic_starts = [
        [1.0, -0.1, 0.01],
        [1.0, -0.05, 0.005],
        [0.5, -0.05, 0.005],
        [1.0, -0.15, 0.015],
        [2.0, -0.2, 0.02],
        [0.1, -0.01, 0.001],
    ]
    local_starts.extend(heuristic_starts)
    
    for x0 in local_starts:
        if time.time() > deadline - 10.0:
            break
        try:
            res = minimize(
                _compute_bound_analytic,
                x0,
                jac=_compute_bound_grad,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50}
            )
            if res.fun < best_fun:
                _update_best(res.fun, res.x)
        except Exception:
            continue
    
    # Phase 3: Nelder-Mead polish on the best point found
    if time.time() < deadline - 5.0:
        try:
            nm_res = minimize(
                _compute_bound_analytic,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            if nm_res.fun < best_fun:
                _update_best(nm_res.fun, nm_res.x)
        except Exception:
            pass
    
    # Phase 4: Final L-BFGS-B polish
    if time.time() < deadline - 2.0:
        try:
            res = minimize(
                _compute_bound_analytic,
                best_x,
                jac=_compute_bound_grad,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 100}
            )
            if res.fun < best_fun:
                _update_best(res.fun, res.x)
        except Exception:
            continue
    
    return np.array(best_x, dtype=np.float64)

# EVOLVE-BLOCK-END
