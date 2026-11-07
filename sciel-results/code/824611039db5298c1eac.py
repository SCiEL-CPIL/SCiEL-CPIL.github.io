"""Optimized program for second autocorrelation inequality."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 140.0  # safety margin before 150s kill

def _compute_value(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    if len(h) != N or np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)  # 99
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)  # spacing: linspace(-0.5, 0.5, m+2) has spacing 1/(m+1)
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    return l2 / (norm_1 * norm_inf)

def _objective(params):
    """Objective to minimize (negative of value) using log-parameterization."""
    # params: log of heights, so h = exp(params) ensures nonnegativity
    h = np.exp(params)
    # Normalize: scale invariance means we can fix sum(h^2) = 1
    # But let's work with the raw value since it's scale-invariant
    return -_compute_value(h)

def _objective_grad(params):
    """Objective and gradient for L-BFGS-B."""
    h = np.exp(params)
    val = _compute_value(h)
    # Numerical gradient
    grad = np.zeros(N)
    eps = 1e-7
    for i in range(N):
        params_p = params.copy()
        params_p[i] += eps
        h_p = np.exp(params_p)
        val_p = _compute_value(h_p)
        grad[i] = (val_p - val) / eps
    return -val, -grad

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Seed: 50 equal heights
    h_seed = np.ones(N)
    best_h = h_seed.copy()
    best_val = _compute_value(best_h)
    
    # Parameterization: h = exp(a) * scale, but since scale-invariant,
    # just use a = log(h) and optimize a
    a0 = np.zeros(N)  # log(1) = 0
    
    # Try L-BFGS-B with numerical gradient
    try:
        result = minimize(
            _objective,
            a0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if time.time() - start_time < DEADLINE:
            a_opt = result.x
            h_opt = np.exp(a_opt)
            val_opt = _compute_value(h_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt
    except Exception:
        pass
    
    # If time remains, try a few random restarts with perturbation
    if time.time() - start_time < DEADLINE - 30:
        rng = np.random.RandomState(42)
        for trial in range(10):
            if time.time() - start_time > DEADLINE - 10:
                break
            # Perturb the current best
            a_curr = np.log(best_h + 1e-10)
            noise = rng.normal(0, 0.3, size=N)
            a_try = a_curr + noise
            h_try = np.exp(a_try)
            
            try:
                res = minimize(
                    _objective,
                    a_try,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if time.time() - start_time > DEADLINE - 5:
                    break
                h_new = np.exp(res.x)
                val_new = _compute_value(h_new)
                if val_new > best_val:
                    best_val = val_new
                    best_h = h_new
            except Exception:
                continue
    
    # Ensure nonnegativity and finiteness
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 1.0)
    
    return list(best_h)
# EVOLVE-BLOCK-END
