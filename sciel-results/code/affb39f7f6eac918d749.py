"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _soft_max(x, tau=1.0):
    """Smooth approximation of max(x) using log-sum-exp."""
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))

def _overlaps(h):
    """Compute overlap for all shifts. overlap[k] = sum_i h[i] * (1 - h[(i+k) % N])"""
    h = np.asarray(h)
    n = len(h)
    overlaps = np.zeros(n)
    for k in range(n):
        shifted = np.roll(h, -k)
        overlaps[k] = np.sum(h * (1.0 - shifted))
    return overlaps

def _objective(params, tau=0.5):
    """Objective: soft-max of overlaps + penalty for constraint violations."""
    # Decode params to h in [0,1] with sum = N/2
    # Use sigmoid-like transform: h[i] = 0.5 + 0.5 * tanh(params[i])
    h = 0.5 + 0.5 * np.tanh(params)
    
    # Compute overlaps
    ov = _overlaps(h)
    
    # Soft max of overlaps
    obj = _soft_max(ov, tau)
    
    # Penalty for sum constraint: sum(h) should be N/2
    sum_h = np.sum(h)
    penalty = 1000.0 * (sum_h - N / 2.0) ** 2
    
    # Penalty for bounds (should be small since tanh maps to [0,1])
    bound_penalty = 1000.0 * (np.sum(np.maximum(0, h - 1.0)) + np.sum(np.maximum(0, -h)))
    
    return obj + penalty + bound_penalty

def _construct_single(tau=0.5, max_iter=500, seed=42):
    """Run a single optimization from a random initialization."""
    rng = np.random.RandomState(seed)
    x0 = rng.randn(N) * 0.5
    
    result = minimize(
        _objective,
        x0,
        args=(tau,),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    h = 0.5 + 0.5 * np.tanh(result.x)
    
    # Project to satisfy constraints exactly
    # Scale to make sum = N/2
    sum_h = np.sum(h)
    if sum_h > 0:
        h = h * (N / 2.0) / sum_h
    
    # Clip to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    
    # Re-adjust sum after clipping
    sum_h = np.sum(h)
    if sum_h != N / 2.0:
        # Simple adjustment: scale non-bound elements
        diff = N / 2.0 - sum_h
        if diff > 0:
            # Need to increase; increase elements not at 1
            mask = h < 1.0
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
        else:
            # Need to decrease; decrease elements not at 0
            mask = h > 0.0
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
    
    h = np.clip(h, 0.0, 1.0)
    
    # Final exact sum fix
    sum_h = np.sum(h)
    if abs(sum_h - N / 2.0) > 1e-10:
        # Distribute the difference evenly
        h += (N / 2.0 - sum_h) / N
        h = np.clip(h, 0.0, 1.0)
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _overlaps(best_h).max()
    
    # Try multiple initializations with different seeds
    seeds = [42, 123, 7, 99, 2024, 555, 111, 314]
    taus = [0.5, 1.0, 2.0, 0.1]
    
    for tau in taus:
        for seed in seeds:
            if time.time() > deadline - 5.0:
                break
            try:
                h = _construct_single(tau=tau, max_iter=300, seed=seed)
                ov = _overlaps(h)
                score = ov.max()
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
            except Exception:
                pass
        if time.time() > deadline - 5.0:
            break
    
    # Final polish: local optimization with tighter tolerance
    if time.time() < deadline - 10.0:
        try:
            params = np.arctanh(2.0 * best_h - 1.0)  # Invert tanh
            # Clip to avoid infinity
            params = np.clip(params, -10, 10)
            
            result = minimize(
                _objective,
                params,
                args=(0.1,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            h = 0.5 + 0.5 * np.tanh(result.x)
            sum_h = np.sum(h)
            if sum_h > 0:
                h = h * (N / 2.0) / sum_h
            h = np.clip(h, 0.0, 1.0)
            
            # Exact sum fix
            sum_h = np.sum(h)
            h += (N / 2.0 - sum_h) / N
            h = np.clip(h, 0.0, 1.0)
            
            ov = _overlaps(h)
            score = ov.max()
            if score < best_score:
                best_score = score
                best_h = h.copy()
        except Exception:
            pass
    
    # Ensure exact sum
    sum_h = np.sum(best_h)
    if abs(sum_h - N / 2.0) > 1e-12:
        best_h += (N / 2.0 - sum_h) / N
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
