"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution

N = 100

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h over all shifts."""
    one_minus_h = 1.0 - h
    max_ov = 0.0
    # Using FFT for O(N log N) speed
    # Overlap at shift s is sum_i h[i] * (1-h)[(i+s)%N]
    # This is a circular cross-correlation.
    # Let's stick to the O(N^2) loop for N=100 as it is simple and fast enough (100*100=10k ops).
    for s in range(N):
        shifted = np.roll(one_minus_h, s)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _construct_h(w, c, sharpness=1.0):
    """
    Construct h given width w and center c.
    h[i] = 0.5 + 0.5 * tanh(sharpness * (i - c) / w)
    """
    i = np.arange(N)
    if w < 1e-6:
        w = 1e-6
    h = 0.5 + 0.5 * np.tanh(sharpness * (i - c) / w)
    return h

def _solve_c(w, sharpness=1.0):
    """
    Find c such that sum(h) = N/2.
    sum(h) is monotonic in c (decreasing).
    """
    target_sum = N / 2.0
    
    lo = -10.0
    hi = float(N) + 10.0
    
    for _ in range(40):
        mid = (lo + hi) / 2.0
        h_mid = _construct_h(w, mid, sharpness)
        s_mid = np.sum(h_mid)
        if s_mid > target_sum:
            lo = mid
        else:
            hi = mid
            
    c_opt = (lo + hi) / 2.0
    return c_opt

def _objective_1d(w_log, sharpness=1.0):
    """
    Objective function for 1D optimization of log(w).
    """
    w = np.exp(w_log)
    if w < 1e-3:
        w = 1e-3
    if w > N:
        w = N
        
    c = _solve_c(w, sharpness)
    h = _construct_h(w, c, sharpness)
    
    # Ensure sum is exactly N/2
    current_sum = np.sum(h)
    diff = N/2.0 - current_sum
    if abs(diff) > 1e-9:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        
    ov = _max_overlap(h)
    return ov

def construct():
    """
    Construct the sequence of step heights.
    """
    # We will search over w (width) and potentially sharpness.
    # Let's fix sharpness to 1.0 (standard tanh) first, as it is a good baseline.
    # If we want to explore other shapes, we can add sharpness to the search space.
    
    # Coarse grid search over log(w)
    log_w_grid = np.linspace(-3.0, 3.0, 50)
    
    best_score = float('inf')
    best_w_log = None
    best_sharpness = 1.0
    
    # Evaluate coarse grid
    for lw in log_w_grid:
        w = np.exp(lw)
        c = _solve_c(w)
        h = _construct_h(w, c)
        current_sum = np.sum(h)
        diff = N/2.0 - current_sum
        if abs(diff) > 1e-9:
            h = h + diff / N
            h = np.clip(h, 0.0, 1.0)
        ov = _max_overlap(h)
        if ov < best_score:
            best_score = ov
            best_w_log = lw
            
    # Refine around the best coarse solution using minimize_scalar
    # Also try a few other candidates to avoid local minima
    
    candidates = []
    for lw in log_w_grid:
        w = np.exp(lw)
        c = _solve_c(w)
        h = _construct_h(w, c)
        current_sum = np.sum(h)
        diff = N/2.0 - current_sum
        if abs(diff) > 1e-9:
            h = h + diff / N
            h = np.clip(h, 0.0, 1.0)
        ov = _max_overlap(h)
        candidates.append((ov, lw))
        
    candidates.sort()
    
    # Refine top 5
    for ov, lw in candidates[:5]:
        lower_bound = max(-3.5, lw - 0.5)
        upper_bound = min(3.5, lw + 0.5)
        
        res = minimize_scalar(_objective_1d, bounds=(lower_bound, upper_bound), method='bounded', options={'xatol': 1e-6})
        
        if res.fun < best_score:
            best_score = res.fun
            best_w_log = res.x
            
    # Final construction
    best_w = np.exp(best_w_log)
    c = _solve_c(best_w)
    h = _construct_h(best_w, c)
    
    # Final correction for exact sum
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    
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
