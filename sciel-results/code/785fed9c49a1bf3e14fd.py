# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

_DEADLINE = 600.0
_MARGIN = 20.0
_START = time.time()

def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START)

def _P_for_support(s, w):
    """Exact P[X1+X2+X3 < 2*X4] in float64."""
    s = np.asarray(s, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n = s.size
    if n == 0 or n != w.size:
        return -np.inf
    if np.any(np.isnan(s)) or np.any(np.isinf(s)) or np.any(np.isnan(w)) or np.any(np.isinf(w)):
        return -np.inf
    if np.any(s < -1e-9) or np.any(w < -1e-9):
        return -np.inf
    tot = w.sum()
    if tot <= 1e-9:
        return -np.inf
    p = w / tot
    pair_sums = (s[:, None] + s[None, :]).ravel()
    pair_probs = (p[:, None] * p[None, :]).ravel()
    order = np.argsort(pair_sums, kind='mergesort')
    sorted_sums = pair_sums[order]
    cum = np.cumsum(pair_probs[order])
    thr = (2.0 * s[None, :] - s[:, None]).ravel()
    idx = np.searchsorted(sorted_sums, thr, side='left')
    F = np.zeros_like(idx)
    m = idx > 0
    F[m] = cum[idx[m] - 1]
    F = F.reshape(n, n)
    return float(np.sum((p[:, None] * p[None, :]) * F))

def _two_point():
    p = (15.0 - np.sqrt(33.0)) / 16.0
    s = np.array([0.0, 1.0], dtype=np.float64)
    w = np.array([p, 1.0 - p], dtype=np.float64)
    return s, w

def _mixture_model(params, n):
    """
    Model: Mass p0 at 0, remaining mass (1-p0) distributed over n-1 Pareto quantiles.
    params: [log_p0, log_alpha, log_xm]
    - p0: probability mass at 0
    - alpha: Pareto shape parameter
    - xm: Pareto scale parameter
    """
    log_p0, log_alpha, log_xm = params
    p0 = float(np.exp(log_p0))
    alpha = float(np.exp(log_alpha))
    xm = float(np.exp(log_xm))
    
    # Clamp parameters to valid ranges
    p0 = min(max(p0, 0.01), 0.95)
    alpha = min(max(alpha, 1.05), 10.0)
    xm = min(max(xm, 0.01), 100.0)

    # Support points: s[0] = 0
    # s[1:] are quantiles of the Pareto distribution
    # We want to sample the "body" of the distribution, avoiding the extreme tail which might be too large
    # Use u in [0.01, 0.99] to avoid infinities and extreme values
    if n > 1:
        u = np.linspace(0.01, 0.99, n - 1)
        # Pareto quantile: Q(u) = xm * (1 - u)^(-1/alpha)
        s_tail = xm * (1.0 - u) ** (-1.0 / alpha)
        s = np.concatenate(([0.0], s_tail))
    else:
        s = np.array([0.0])

    # Weights
    w = np.zeros(n)
    w[0] = p0
    if n > 1:
        # Distribute remaining mass uniformly over the quantiles
        # This is a common choice for discretizing a continuous distribution
        w[1:] = (1.0 - p0) / (n - 1)
    
    s = np.maximum(s, 0.0)
    return s, w

def _objective(params, n):
    s, w = _mixture_model(params, n)
    P = _P_for_support(s, w)
    if P == -np.inf:
        return 1.0
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    if _time_left() <= 0.0:
        return _two_point()

    best_P = -np.inf
    best_s = None
    best_w = None

    def _consider(s, w):
        nonlocal best_P, best_s, best_w
        s = np.asarray(s, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        if s.size == 0 or s.size != w.size:
            return
        if np.any(np.isnan(s)) or np.any(np.isinf(s)) or np.any(np.isnan(w)) or np.any(np.isinf(w)):
            return
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return
        if w.sum() <= 1e-9:
            return
        P = _P_for_support(s, w)
        if P != -np.inf and P > best_P:
            best_P, best_s, best_w = P, s.copy(), w.copy()

    # ---- 2-point baseline ----
    s2, w2 = _two_point()
    P2 = _P_for_support(s2, w2)
    if P2 > best_P:
        best_P, best_s, best_w = P2, s2.copy(), w2.copy()

    # ---- Mixture Model Optimization ----
    # Parameters: log_p0, log_alpha, log_xm
    # Ranges:
    # p0 in [0.01, 0.95] -> log_p0 in [log(0.01), log(0.95)]
    # alpha in [1.05, 10.0] -> log_alpha in [log(1.05), log(10.0)]
    # xm in [0.01, 100.0] -> log_xm in [log(0.01), log(100.0)]
    
    bounds = [
        (np.log(0.01), np.log(0.95)),
        (np.log(1.05), np.log(10.0)),
        (np.log(0.01), np.log(100.0))
    ]

    # Try different n values
    for n in [8, 16, 32, 64]:
        if _time_left() <= 60.0:
            break
            
        # Coarse grid search to find a good region
        if _time_left() > 30.0:
            p0_vals = [0.2, 0.4, 0.6, 0.8]
            alpha_vals = [1.5, 2.0, 3.0, 5.0]
            xm_vals = [0.5, 1.0, 2.0, 5.0]
            
            for p0 in p0_vals:
                for alpha in alpha_vals:
                    for xm in xm_vals:
                        if _time_left() <= 5.0:
                            break
                        params = [np.log(p0), np.log(alpha), np.log(xm)]
                        s, w = _mixture_model(params, n)
                        _consider(s, w)
        
        # Local optimization from best found so far or random starts
        if _time_left() > 20.0:
            # Use Differential Evolution for global search in this parameter space
            try:
                res = differential_evolution(
                    _objective,
                    bounds,
                    args=(n,),
                    seed=42,
                    maxiter=200,
                    popsize=10,
                    tol=1e-6,
                    mutation=(0.5, 1.2),
                    recombination=0.8,
                    polish=True,
                )
                P_val = -res.fun
                if P_val > best_P:
                    s, w = _mixture_model(res.x, n)
                    _consider(s, w)
            except Exception:
                pass

    # ---- Final Polish ----
    if _time_left() > 10.0 and best_s is not None:
        # Try to polish the best solution found with Nelder-Mead
        # We need to infer params from best_s, best_w or just use the last res.x if available
        # Since we don't store the params that produced best_s, we can't directly polish it easily without re-optimizing.
        # However, we can run a final DE with tighter bounds around the best known P if we had stored params.
        # For now, we rely on the DE polish step inside the loop.
        pass

    if best_s is None:
        return _two_point()

    # Final validity check
    if len(best_s) != len(best_w):
        return _two_point()
    if np.any(best_s < -1e-9) or np.any(best_w < -1e-9) or best_w.sum() <= 1e-9:
        return _two_point()
    if np.any(np.isnan(best_s)) or np.any(np.isinf(best_s)) or np.any(np.isnan(best_w)) or np.any(np.isinf(best_w)):
        return _two_point()
        
    return best_s, best_w

# EVOLVE-BLOCK-END
