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

def _construct_h_linear(w, c):
    """
    Construct h given width w and center c.
    Linear ramp from 0 to 1 over [c-w/2, c+w/2].
    h[i] = clip((i - (c - w/2)) / w, 0, 1)
    """
    i = np.arange(N)
    if w < 1e-6:
        # Step function
        # h[i] = 0 if i < c else 1
        # Need to handle discretization carefully for sum
        h = np.where(i < c, 0.0, 1.0)
        return h
    
    start = c - w / 2.0
    end = c + w / 2.0
    
    # Calculate linear ramp
    # (i - start) / w
    raw = (i - start) / w
    h = np.clip(raw, 0.0, 1.0)
    return h

def _solve_c_linear(w):
    """
    Find c such that sum(h) = N/2.
    """
    target_sum = N / 2.0
    
    def sum_diff(c):
        h = _construct_h_linear(w, c)
        return np.sum(h) - target_sum
    
    lo = 0.0
    hi = float(N)
    
    # sum is decreasing with c
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_mid = _construct_h_linear(w, mid)
        s_mid = np.sum(h_mid)
        if s_mid > target_sum:
            lo = mid
        else:
            hi = mid
            
    c_opt = (lo + hi) / 2.0
    return c_opt

def _objective_1d_linear(w_log):
    """
    Objective function for 1D optimization of log(w).
    """
    w = np.exp(w_log)
    if w < 1e-3:
        w = 1e-3
    if w > N:
        w = N
        
    c = _solve_c_linear(w)
    h = _construct_h_linear(w, c)
    
    current_sum = np.sum(h)
    if abs(current_sum - N/2.0) > 1e-6:
        diff = N/2.0 - current_sum
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        
    return _max_overlap(h)

def construct():
    """
    Construct the sequence of step heights.
    """
    # Optimize log(w)
    # w ranges from ~1 to ~100
    res = minimize_scalar(_objective_1d_linear, bounds=(0.0, np.log(N)), method='bounded', options={'xatol': 1e-4})
    
    best_w_log = res.x
    best_w = np.exp(best_w_log)
    
    c = _solve_c_linear(best_w)
    h = _construct_h_linear(best_w, c)
    
    # Final correction for exact sum
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    
    if abs(diff) > 1e-10:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        
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
