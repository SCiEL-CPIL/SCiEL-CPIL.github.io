# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE_MARGIN = 20.0  # seconds to spare before the 210s kill

def _remaining():
    return _START_TIME + 210.0 - _DEADLINE_MARGIN - time.time()

# Pre-compute Gauss-Hermite nodes and weights
_N_QUAD = 100
_nodes, _weights = np.polynomial.hermite.hermgauss(_N_QUAD)

def _eval_hermite(x, n):
    if n == 0: return np.ones_like(x)
    if n == 4: return 16*x**4 - 48*x**2 + 12
    if n == 8: return 256*x**8 - 4480*x**6 + 26880*x**4 - 53760*x**2 + 1680
    if n == 12: return 4096*x**12 - 147456*x**10 + 2419200*x**8 - 16329600*x**6 + 43224000*x**4 - 40828800*x**2 + 665280
    raise ValueError("Unsupported degree")

_H0_nodes = _eval_hermite(_nodes, 0)
_H4_nodes = _eval_hermite(_nodes, 4)
_H8_nodes = _eval_hermite(_nodes, 8)
_H12_nodes = _eval_hermite(_nodes, 12)

_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 665280.0

# Precompute complex evaluations
_i_nodes = 1j * _nodes
_H0_i = _eval_hermite(_i_nodes, 0)
_H4_i = _eval_hermite(_i_nodes, 4)
_H8_i = _eval_hermite(_i_nodes, 8)
_H12_i = _eval_hermite(_i_nodes, 12)

def _compute_upper_bound(c0, c4, c8):
    c12 = -(c0 * _H0_0 + c4 * _H4_0 + c8 * _H8_0) / _H12_0
    
    P_nodes = c0 * _H0_nodes + c4 * _H4_nodes + c8 * _H8_nodes + c12 * _H12_nodes
    norm_sq = np.sum(P_nodes**2 * _weights)
    
    P_i_nodes = c0 * _H0_i + c4 * _H4_i + c8 * _H8_i + c12 * _H12_i
    abs2_P_i = np.abs(P_i_nodes)**2
    norm_f_sq = 2 * np.pi * np.sum(abs2_P_i * _weights)
    
    return norm_sq * norm_f_sq

def construct():
    t_start = time.time()
    deadline = t_start + 210.0 - _DEADLINE_MARGIN
    
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best = seed.copy()
    best_val = _compute_upper_bound(best[0], best[1], best[2])
    
    def objective(params):
        if time.time() > deadline:
            return 1e30
        return _compute_upper_bound(params[0], params[1], params[2])
    
    # Phase 1: Global search with differential evolution
    if time.time() < deadline - 60.0:
        try:
            bounds = [(-5.0, 5.0), (-0.5, 0.5), (-0.01, 0.01)]
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-10,
                seed=42,
                polish=False
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement from multiple starting points
    start_points = [
        best,
        seed,
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-0.5, 0.02, 0.0001]),
        np.array([-1.5, 0.08, 0.001]),
        np.array([-0.8, 0.03, 0.0002]),
        np.array([-2.0, 0.1, 0.002]),
        np.array([-0.3, 0.01, 0.00005]),
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                objective,
                sp,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50}
            )
            if result.fun < best_val:
                best_val = result.fun
                best = result.x.copy()
        except Exception:
            continue
    
    # Phase 3: Final high-precision refinement
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                objective,
                best,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-18, 'maxls': 100}
            )
            if result.fun < best_val:
                best_val = result.fun
                best = result.x.copy()
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
