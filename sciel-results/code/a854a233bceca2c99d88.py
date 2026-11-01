"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
import time

_DEADLINE = 200.0  # safety margin before 210s kill
_start = time.time()

def _elapsed():
    return time.time() - _start

def _hermite_basis(x, max_deg=12):
    """Compute normalized Hermite polynomials H_0..H_max_deg at points x.
    Using probabilist's (physicist's?) - we use scipy eval_hermite (physicist's)."""
    n = max_deg // 2 + 1
    coefs = []
    for k in range(0, max_deg + 1, 4):
        coefs.append(eval_hermite(k)(x))
    return coefs

def _g_of_x(x, a0, a4, a8, a12):
    """g(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12, times Gaussian (absorbed into normalization)."""
    return a0 * eval_hermite(0)(x) + a4 * eval_hermite(4)(x) + a8 * eval_hermite(8)(x) + a12 * eval_hermite(12)(x)

def _a12_from_g0(a0, a4, a8):
    """Solve g(0) = 0 for a12. H_k(0) is known."""
    H0_0 = eval_hermite(0)(0.0)
    H4_0 = eval_hermite(4)(0.0)
    H8_0 = eval_hermite(8)(0.0)
    H12_0 = eval_hermite(12)(0.0)
    # a0*H0_0 + a4*H4_0 + a8*H8_0 + a12*H12_0 = 0
    if abs(H12_0) < 1e-15:
        return 0.0
    return -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0

def _fourier_transform_of_hermite_combo(x, a0, a4, a8, a12):
    """FT of H_k(x) e^{-x^2/2} is (-1)^k H_k(x) e^{-x^2/2} (up to 2pi factors, cancel in ratio).
    So |ĝ| has same magnitude as |g| if we use normalized basis. But the actual
    sup|g| and sup|ĝ| may differ due to the Gaussian envelope.
    
    Actually for the Hermite-Gaussian: if g(x) = sum a_k H_k(x) e^{-x^2/2},
    then ĝ(x) = sum a_k (-1)^k H_k(x) e^{-x^2/2} (in appropriate normalization).
    
    So sup|ĝ| = sup|sum a_k (-1)^k H_k(x) e^{-x^2/2}|
    """
    return (a0 * eval_hermite(0)(x) + a4 * (-1)**4 * eval_hermite(4)(x) + 
            a8 * (-1)**8 * eval_hermite(8)(x) + a12 * (-1)**12 * eval_hermite(12)(x)) * np.exp(-x**2 / 2.0)

def _g_with_gaussian(x, a0, a4, a8, a12):
    """g(x) = [a0 H0 + a4 H4 + a8 H8 + a12 H12](x) * exp(-x^2/2)"""
    poly = (a0 * eval_hermite(0)(x) + a4 * eval_hermite(4)(x) + 
            a8 * eval_hermite(8)(x) + a12 * eval_hermite(12)(x))
    return poly * np.exp(-x**2 / 2.0)

def _score_params(a0, a4, a8):
    """Compute upper bound: sup|g| * sup|g_hat|"""
    a12 = _a12_from_g0(a0, a4, a8)
    
    # Evaluate on a grid
    x = np.linspace(-15, 15, 20001)
    g_vals = _g_with_gaussian(x, a0, a4, a8, a12)
    gh_vals = _fourier_transform_of_hermite_combo(x, a0, a4, a8, a12)
    
    sup_g = np.max(np.abs(g_vals))
    sup_gh = np.max(np.abs(gh_vals))
    
    if sup_g < 1e-15 or sup_gh < 1e-15:
        return 0.0
    
    return sup_g * sup_gh

def _local_search(x0, max_iter=500, step=0.1):
    """Simple gradient-free local search (Nelder-Mead style or coordinate descent)."""
    from scipy.optimize import minimize
    
    def objective(v):
        if _elapsed() > _DEADLINE - 5:
            return 1e10
        return _score_params(v[0], v[1], v[2])
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8})
        if _elapsed() > _DEADLINE - 5:
            return x0, _score_params(*x0)
        if res.success or res.fun < _score_params(*x0):
            return res.x, res.fun
        return x0, _score_params(*x0)
    except Exception:
        return x0, _score_params(*x0)

def construct():
    best_x = np.array([-1.13, 0.04, 0.000308642])  # seed values
    best_score = _score_params(*best_x)
    
    # Multi-start local search
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([-1.0, 0.05, 0.001]),
        np.array([-1.2, 0.03, 0.0001]),
        np.array([-0.9, 0.06, 0.002]),
        np.array([-1.1, 0.045, 0.0005]),
        np.array([-1.15, 0.035, 0.0002]),
        np.array([-1.05, 0.042, 0.0008]),
        np.array([-1.25, 0.03, 0.00015]),
    ]
    
    for start in starts:
        if _elapsed() > _DEADLINE - 10:
            break
        x_opt, s = _local_search(start, max_iter=300)
        if s < best_score:
            best_score = s
            best_x = x_opt
    
    # Additional refinement with more iterations if time permits
    if _elapsed() < _DEADLINE - 20:
        x_opt, s = _local_search(best_x, max_iter=1000)
        if s < best_score:
            best_score = s
            best_x = x_opt
    
    return np.array([best_x[0], best_x[1], best_x[2]])
# EVOLVE-BLOCK-END
