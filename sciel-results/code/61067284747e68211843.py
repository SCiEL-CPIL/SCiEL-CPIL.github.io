"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 100

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h over all shifts."""
    one_minus_h = 1.0 - h
    max_ov = 0.0
    # Using np.roll for each shift is O(N^2) total, which is fine for N=100.
    # We can also use FFT for O(N log N) but N is small.
    for s in range(N):
        shifted = np.roll(one_minus_h, s)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _construct_h(w, c):
    """
    Construct h given width w and center c.
    h[i] = 0.5 + 0.5 * tanh((i - c) / w)
    """
    i = np.arange(N)
    # Avoid division by zero
    if w < 1e-6:
        w = 1e-6
    h = 0.5 + 0.5 * np.tanh((i - c) / w)
    return h

def _solve_c(w):
    """
    Find c such that sum(h) = N/2.
    sum(h) is monotonic in c.
    """
    target_sum = N / 2.0
    
    # Binary search for c in [0, N]
    # sum(h) is decreasing with c based on previous analysis.
    # Let's verify bounds.
    # If c is very negative, i-c is large positive, tanh ~ 1, h ~ 1, sum ~ N.
    # If c is very positive, i-c is large negative, tanh ~ -1, h ~ 0, sum ~ 0.
    # So we search for c where sum crosses N/2.
    
    # We can estimate the range of c.
    # For a step function at c, sum is roughly N/2 if c is in the middle.
    # Let's search c in [-10, N+10] to be safe.
    
    lo = -10.0
    hi = float(N) + 10.0
    
    # Check endpoints
    sum_lo = np.sum(_construct_h(w, lo))
    sum_hi = np.sum(_construct_h(w, hi))
    
    # If the target is not between sum_lo and sum_hi, return the closer one?
    # But theoretically, for any w, we can find a c.
    # sum_lo ~ N, sum_hi ~ 0. Target N/2 is in between.
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_mid = _construct_h(w, mid)
        s_mid = np.sum(h_mid)
        if s_mid > target_sum:
            lo = mid
        else:
            hi = mid
            
    c_opt = (lo + hi) / 2.0
    return c_opt

def _evaluate_params(w, c):
    """
    Evaluate the objective for given w and c.
    Returns (overlap, h)
    """
    h = _construct_h(w, c)
    # Ensure sum is exactly N/2 by small adjustment if needed?
    # The binary search should be accurate enough.
    current_sum = np.sum(h)
    if abs(current_sum - N/2.0) > 1e-6:
        diff = N/2.0 - current_sum
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        
    ov = _max_overlap(h)
    return ov, h

def _objective_1d(w_log):
    """
    Objective function for 1D optimization of log(w).
    """
    w = np.exp(w_log)
    # Clamp w to reasonable range
    if w < 1e-3:
        w = 1e-3
    if w > N:
        w = N
        
    c = _solve_c(w)
    ov, _ = _evaluate_params(w, c)
    return ov

def construct():
    """
    Construct the sequence of step heights.
    """
    # Strategy:
    # 1. Coarse grid search over log(w) to find good regions.
    # 2. Local refinement using minimize_scalar on the best candidates.
    
    # Log range for w: w from 0.01 to 100.
    # log(0.01) ~ -4.6, log(100) ~ 4.6.
    
    # Coarse grid
    log_w_grid = np.linspace(-4.0, 4.0, 100)
    
    best_score = float('inf')
    best_w_log = None
    best_c = None
    
    # Evaluate coarse grid
    for lw in log_w_grid:
        w = np.exp(lw)
        if w < 1e-3 or w > N:
            continue
        c = _solve_c(w)
        ov, _ = _evaluate_params(w, c)
        if ov < best_score:
            best_score = ov
            best_w_log = lw
            best_c = c
            
    # Refine around the best coarse solution
    # Also check a few other top candidates to avoid local minima
    
    # Get top 5 candidates from grid
    # Re-evaluate to get scores
    candidates = []
    for lw in log_w_grid:
        w = np.exp(lw)
        if w < 1e-3 or w > N:
            continue
        c = _solve_c(w)
        ov, _ = _evaluate_params(w, c)
        candidates.append((ov, lw))
        
    candidates.sort()
    
    # Refine top 3
    for ov, lw in candidates[:3]:
        # Define bounds for local search around lw
        lower_bound = max(-4.6, lw - 0.5)
        upper_bound = min(4.6, lw + 0.5)
        
        res = minimize_scalar(_objective_1d, bounds=(lower_bound, upper_bound), method='bounded', options={'xatol': 1e-5})
        
        if res.fun < best_score:
            best_score = res.fun
            best_w_log = res.x
            best_c = _solve_c(np.exp(best_w_log))
            
    # Final construction
    best_w = np.exp(best_w_log)
    c = best_c
    
    h = _construct_h(best_w, c)
    
    # Final correction for exact sum
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    
    # Distribute difference
    if abs(diff) > 1e-10:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        
        # Iterative fix for exact sum
        for _ in range(20):
            current_sum = np.sum(h)
            diff = N / 2.0 - current_sum
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                mask = h < 1.0
                if not np.any(mask): break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                for idx in indices:
                    h[idx] = min(1.0, h[idx] + step)
            else:
                mask = h > 0.0
                if not np.any(mask): break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                for idx in indices:
                    h[idx] = max(0.0, h[idx] + step)

    return h
