"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution

N = 100

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h over all shifts."""
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for s in range(N):
        shifted = np.roll(one_minus_h, s)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _construct_h(w, c):
    """
    Construct h given width w and center c.
    Using a sigmoid-like transition.
    h[i] = 1 / (1 + exp(-(i - c) / w))
    """
    i = np.arange(N)
    if w < 1e-6:
        w = 1e-6
    # Sigmoid
    h = 1.0 / (1.0 + np.exp(-(i - c) / w))
    return h

def _solve_c(w):
    """
    Find c such that sum(h) = N/2.
    """
    target_sum = N / 2.0
    
    def sum_diff(c):
        h = _construct_h(w, c)
        return np.sum(h) - target_sum
    
    # Bisection method
    # sum(h) is decreasing with c for sigmoid?
    # i < c -> (i-c)/w < 0 -> exp(pos) > 1 -> h < 0.5
    # i > c -> (i-c)/w > 0 -> exp(neg) < 1 -> h > 0.5
    # So h is 0 on left, 1 on right.
    # As c increases, the step moves right.
    # More indices are on the left (h~0), fewer on right (h~1).
    # So sum decreases as c increases.
    
    lo = -10.0 * w
    hi = N + 10.0 * w
    
    # Ensure we bracket the root
    # If sum(lo) < target, we need smaller c (larger sum)? No, sum decreases with c.
    # So if sum(lo) < target, we need to decrease c?
    # Wait.
    # c = -10w. i=0..N. i-c is large positive. h ~ 1. Sum ~ N.
    # c = N+10w. i-c is large negative. h ~ 0. Sum ~ 0.
    # So sum(lo) ~ N > target. sum(hi) ~ 0 < target.
    # We want sum = target.
    # Since sum is decreasing, if sum(mid) > target, we need to increase c.
    # If sum(mid) < target, we need to decrease c.
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        s_mid = np.sum(_construct_h(w, mid))
        if s_mid > target_sum:
            lo = mid
        else:
            hi = mid
            
    c_opt = (lo + hi) / 2.0
    return c_opt

def _objective_1d(w_log):
    """
    Objective function for 1D optimization of log(w).
    """
    w = np.exp(w_log)
    # Clamp w
    if w < 0.1:
        w = 0.1
    if w > 50.0:
        w = 50.0
        
    c = _solve_c(w)
    h = _construct_h(w, c)
    
    # Correct sum
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    if abs(diff) > 1e-9:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        # Re-normalize if clipping caused sum error
        # This is a minor fix, bisection should be good
        current_sum = np.sum(h)
        if abs(current_sum - N/2.0) > 1e-9:
             # Force exact sum by adjusting the middle element or distributing
             # For robustness, just return a high penalty if sum is off
             pass

    return _max_overlap(h)

def construct():
    """
    Construct the sequence of step heights.
    """
    # Global search using differential evolution or grid + refine
    # Let's do a grid search on log(w) first
    log_w_min = np.log(0.1)
    log_w_max = np.log(50.0)
    
    # Coarse grid
    w_grid = np.linspace(log_w_min, log_w_max, 200)
    best_score = float('inf')
    best_w_log = 0.0
    
    for lw in w_grid:
        try:
            score = _objective_1d(lw)
            if score < best_score:
                best_score = score
                best_w_log = lw
        except Exception:
            continue
            
    # Refine around best_w_log
    # Use minimize_scalar on a small window
    delta = (w_grid[1] - w_grid[0]) * 2
    bounds = (best_w_log - delta, best_w_log + delta)
    
    try:
        res = minimize_scalar(_objective_1d, bounds=bounds, method='bounded', options={'xatol': 1e-6})
        if res.success and res.fun < best_score:
            best_w_log = res.x
            best_score = res.fun
    except Exception:
        pass
        
    best_w = np.exp(best_w_log)
    c = _solve_c(best_w)
    h = _construct_h(best_w, c)
    
    # Final exact sum correction
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    
    if abs(diff) > 1e-12:
        # Adjust h to have exact sum N/2
        # Distribute diff evenly, then clip, then fix residual
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        
        # Iterative fix for exact sum
        for _ in range(50):
            current_sum = np.sum(h)
            diff = N / 2.0 - current_sum
            if abs(diff) < 1e-14:
                break
            if diff > 0:
                mask = h < 1.0
                if not np.any(mask): 
                    # Should not happen if N/2 < N
                    break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                for idx in indices:
                    h[idx] = min(1.0, h[idx] + step)
            else:
                mask = h > 0.0
                if not np.any(mask):
                    break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                for idx in indices:
                    h[idx] = max(0.0, h[idx] + step)

    return h
