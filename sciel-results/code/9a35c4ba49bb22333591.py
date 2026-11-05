import numpy as np
from scipy.optimize import minimize
import time

N = 400
_DEADLINE = 150.0
_START_TIME = None


def _objective_hard(x):
    """Compute the actual evaluation value."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    return abs(2 * N * m / (s * s))


def _random_walk_heights(drift, sigma, seed=42):
    """Generate step heights as random walk with negative drift, then center."""
    rng = np.random.RandomState(seed)
    increments = rng.randn(N) * sigma + drift
    h = np.cumsum(increments)
    # Center to remove DC component (mean becomes 0)
    h = h - np.mean(h)
    return h


def _rw_objective(params):
    """Objective for random walk parameterization: [drift, log_sigma, seed_idx]"""
    drift = params[0]
    log_sigma = params[1]
    seed_idx = int(round(params[2])) if len(params) > 2 else 0
    sigma = np.exp(log_sigma)
    
    h = _random_walk_heights(drift, sigma, seed=1000 + seed_idx)
    return _objective_hard(h)


def _optimize_rw(params0, deadline):
    """Optimize random walk parameters using L-BFGS-B."""
    try:
        def obj(x):
            if time.time() > deadline:
                return 1e10
            return _rw_objective(x)
        
        res = minimize(
            obj,
            params0,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 200}
        )
        return res.x, _rw_objective(res.x)
    except Exception:
        return params0, _rw_objective(params0)


def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    best_val = 2.0
    best_h = np.ones(N)
    
    deadline = _START_TIME + _DEADLINE - 5.0
    
    # Strategy 1: Random walk parameterization with multiple seeds
    # Parameters: [drift, log_sigma, seed_idx]
    # We'll try various drift values and let optimizer refine
    
    rw_configs = []
    
    # Different drift values (negative drift for zero-sum structure)
    for drift in [-0.01, -0.02, -0.05, -0.1, -0.2, -0.3, -0.5]:
        for log_sigma in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for seed_idx in range(5):
                rw_configs.append([drift, log_sigma, float(seed_idx)])
    
    # Also try some positive drift for variety
    for drift in [0.01, 0.05, 0.1]:
        for log_sigma in [-0.5, 0.0, 0.5]:
            for seed_idx in range(3):
                rw_configs.append([drift, log_sigma, float(seed_idx)])
    
    # Optimize each config
    for config in rw_configs:
        if time.time() > deadline - 3:
            break
        
        params0 = np.array(config, dtype=float)
        opt_params, val = _optimize_rw(params0, deadline)
        
        if val < best_val:
            best_val = val
            # Reconstruct the heights from optimized parameters
            drift = opt_params[0]
            log_sigma = opt_params[1]
            seed_idx = int(round(opt_params[2])) if len(opt_params) > 2 else 0
            sigma = np.exp(log_sigma)
            h = _random_walk_heights(drift, sigma, seed=1000 + seed_idx)
            best_h = h.copy()
    
    # Strategy 2: Keep the original soft-max optimization but start from best RW result
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize as _minimize
            
            def _soft_max_obj(x, temp=0.05):
                h = x
                s = np.sum(h)
                if abs(s) < 1e-12:
                    return 1e10
                conv = np.convolve(h, h, mode='full')
                m = np.max(conv)
                exp_terms = np.exp((conv - m) / temp)
                logsumexp = m / temp + np.log(np.sum(exp_terms))
                smax = temp * logsumexp
                return smax / (s * s)
            
            def _soft_max_grad(x, temp=0.05):
                h = x
                s = np.sum(h)
                if abs(s) < 1e-12:
                    return np.zeros_like(x)
                
                conv = np.convolve(h, h, mode='full')
                m = np.max(conv)
                exp_terms = np.exp((conv - m) / temp)
                softmax_weights = exp_terms / np.sum(exp_terms)
                sw = softmax_weights
                
                grad = np.zeros(N)
                for i in range(N):
                    max_j = min(N, 2 * N - 1 - i)
                    if max_j > 0:
                        grad[i] += np.dot(sw[i:i + max_j], h[:max_j])
                    
                    max_j2 = min(i + 1, N)
                    if max_j2 > 0:
                        sw_slice = sw[i - max_j2 + 1 : i + 1][::-1]
                        grad[i] += np.dot(sw_slice, h[:max_j2])
                
                smax = temp * (m / temp + np.log(np.sum(exp_terms)))
                d_obj = grad / (s * s) - 2 * smax / (s * s * s)
                return d_obj
            
            # Polish from the best RW result
            res = _minimize(
                lambda x: _soft_max_obj(x, 0.01),
                best_h,
                jac=lambda x: _soft_max_grad(x, 0.01),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 800}
            )
            
            new_val = _objective_hard(res.x)
            if new_val < best_val:
                best_val = new_val
                best_h = res.x.copy()
                
        except Exception:
            pass
    
    # Final quick polish if time remains
    if time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _soft_max_obj(x, 0.005),
                best_h,
                jac=lambda x: _soft_max_grad(x, 0.005),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-9, 'maxfun': 300}
            )
            val = _objective_hard(res.x)
            if val < best_val:
                best_val = val
                best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h
