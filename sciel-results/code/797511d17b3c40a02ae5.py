import numpy as np
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def _soft_max_corr(h, beta=10.0):
    """
    Compute the soft-max of the full correlation between h and 1-h.
    This serves as a smooth surrogate for max(corr).
    """
    corr = np.correlate(h, 1.0 - h, mode='full')
    # log-sum-exp for numerical stability
    max_corr = np.max(corr)
    if max_corr > 1e6: # unlikely but safe
        return max_corr
    shifted = beta * (corr - max_corr)
    sum_exp = np.sum(np.exp(shifted))
    return max_corr + np.log(sum_exp) / beta

def _soft_max_corr_grad(h, beta=10.0):
    """
    Compute the gradient of the soft-max correlation objective.
    """
    corr = np.correlate(h, 1.0 - h, mode='full')
    max_corr = np.max(corr)
    shifted = beta * (corr - max_corr)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp # softmax weights w_m
    
    grad = np.zeros(N)
    # C_m = sum_i h_i (1 - h_{i + m - N + 1})
    # dC_m/dh_j = (1 - h_{j + m - N + 1}) - h_{j - m + N - 1} (with bounds)
    
    # Vectorized computation of gradient
    # For each m, we need to accumulate contributions to all j.
    # This is O(N^2) if done naively. N=100 is small.
    for m in range(len(corr)):
        w = weights[m]
        if w < 1e-12:
            continue
        offset = m - (N - 1)
        # dC_m/dh_j involves h_{j+offset} and h_{j-offset}
        # Let's compute dC_m vector for all j at once
        dCm = np.zeros(N)
        
        # Term 1: (1 - h_{j + offset}) for valid j
        # j + offset in [0, N-1] => j in [-offset, N-1-offset]
        start1 = max(0, -offset)
        end1 = min(N, N - offset)
        if start1 < end1:
            # indices in h for the second factor: j + offset
            idx_h2 = np.arange(start1 + offset, end1 + offset)
            dCm[start1:end1] += (1.0 - h[idx_h2])
            
        # Term 2: -h_{j - offset} for valid j
        # j - offset in [0, N-1] => j in [offset, N-1+offset]
        start2 = max(0, offset)
        end2 = min(N, N + offset)
        if start2 < end2:
            # indices in h for the first factor: j - offset
            idx_h1 = np.arange(start2 - offset, end2 - offset)
            dCm[start2:end2] -= h[idx_h1]
            
        grad += beta * w * dCm
        
    return grad

def _project(h):
    """Project h onto feasible region: h in [0,1]^N, sum(h) = N/2."""
    h = h.copy()
    h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    target = N / 2.0
    diff = target - s
    if abs(diff) > 1e-12:
        mask = (h > 1e-6) & (h < 1.0 - 1e-6)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            # Final fix
            s = np.sum(h)
            diff = target - s
            if abs(diff) > 1e-12:
                # Add to any element not at boundary
                mask2 = (h < 1.0) & (h > 0.0)
                if np.any(mask2):
                    h[mask2] += diff / np.sum(mask2)
                    h = np.clip(h, 0.0, 1.0)
                else:
                    # If all saturated, just clip and hope sum is close enough
                    # This case is rare with random starts
                    pass
    return h

def _optimize_h(h0, beta=20.0, max_iter=200):
    """
    Optimize h using SLSQP on the soft-max objective.
    """
    def objective(h):
        return _soft_max_corr(h, beta)
        
    def gradient(h):
        return _soft_max_corr_grad(h, beta)
        
    def constraint_sum(h):
        return np.sum(h) - N / 2.0
        
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    
    try:
        res = minimize(objective, h0, method='SLSQP', jac=gradient, 
                       bounds=bounds, constraints=constraints, 
                       options={'maxiter': max_iter, 'ftol': 1e-12})
        if res.success:
            return res.x, _soft_max_corr(res.x, beta)
        else:
            # Even if not perfect success, res.x might be better
            h_res = _project(res.x)
            val = _soft_max_corr(h_res, beta)
            return h_res, val
    except Exception:
        return h0, _soft_max_corr(h0, beta)

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - SAFETY_MARGIN)
    
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _soft_max_corr(h0, 20.0)
    
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    starts = [h0.copy()]
    
    # Random perturbations
    for scale in [0.1, 0.2, 0.3, 0.5]:
        h_pert = np.full(N, 0.5) + rng.randn(N) * scale
        h_pert = _project(h_pert)
        starts.append(h_pert)
    
    # Structured patterns
    for freq in [1, 2, 3, 4, 5, 8]:
        for amp in [0.2, 0.4, 0.5]:
            x = np.arange(N)
            h_sine = 0.5 + amp * np.sin(2 * np.pi * freq * x / N)
            h_sine = _project(h_sine)
            starts.append(h_sine)
            
    # Alternating
    for a in [0.2, 0.3, 0.4]:
        h_alt = np.where(np.arange(N) % 2 == 0, a, 1.0 - a)
        h_alt = _project(h_alt)
        starts.append(h_alt)

    # Run optimization from each start
    for h_start in starts:
        if time.time() > deadline - 5:
            break
        
        # Use increasing beta for sharper max approximation
        # Start with lower beta for global search, higher for local polish
        h_opt, val = _optimize_h(h_start, beta=10.0, max_iter=100)
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
            
        # Polish with higher beta if time permits
        if time.time() > deadline - 10:
            break
        h_opt2, val2 = _optimize_h(best_h, beta=50.0, max_iter=50)
        if val2 < best_val:
            best_val = val2
            best_h = h_opt2.copy()

    # Final projection to ensure validity
    best_h = _project(best_h)
    
    return best_h
