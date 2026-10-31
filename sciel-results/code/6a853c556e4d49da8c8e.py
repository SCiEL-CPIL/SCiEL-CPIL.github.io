"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import minimize_scalar

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
    
    # Define function to minimize: (sum(h) - target_sum)^2
    def sum_diff(c):
        h = _construct_h(w, c)
        return np.sum(h) - target_sum
    
    # Binary search for c in [0, N]
    # sum(h) increases with c.
    # At c=0, h is shifted left.
    # At c=N, h is shifted right.
    # We need to check the range.
    # tanh((i-c)/w).
    # If c is very small, i-c is positive for most i, tanh ~ 1, sum ~ N.
    # If c is very large, i-c is negative, tanh ~ -1, sum ~ 0.
    # So sum decreases as c increases?
    # Let's check: i=0, c=0 -> tanh(0)=0. i=10, c=0 -> tanh(10/w) ~ 1.
    # So larger i has larger h.
    # If we increase c, the "step" moves to the right.
    # Indices i < c become more negative argument -> tanh ~ -1 -> h ~ 0.
    # Indices i > c become more positive argument -> tanh ~ 1 -> h ~ 1.
    # So as c increases, the "1" region moves right, and "0" region moves left?
    # No.
    # h[i] = 0.5 + 0.5 tanh((i-c)/w).
    # If i < c, (i-c) < 0, tanh < 0, h < 0.5.
    # If i > c, (i-c) > 0, tanh > 0, h > 0.5.
    # So h is 0 on the left of c and 1 on the right of c.
    # If we increase c, the boundary moves right.
    # More indices are on the "left" (h~0) and fewer on the "right" (h~1)?
    # No, the domain is [0, N).
    # If c=0: i>0 is right. Most i are > 0. So h ~ 1. Sum ~ N.
    # If c=N: i<N is left. Most i are < N. So h ~ 0. Sum ~ 0.
    # So sum(h) is decreasing with c.
    
    lo = 0.0
    hi = float(N)
    
    # Check endpoints
    sum_lo = np.sum(_construct_h(w, lo))
    sum_hi = np.sum(_construct_h(w, hi))
    
    # We want sum = 50.
    # If sum_lo < 50 and sum_hi > 50, swap or adjust logic.
    # Based on above: sum_lo ~ N (100), sum_hi ~ 0.
    # So we are looking for c where sum goes from 100 down to 0.
    
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
    h = _construct_h(w, c)
    
    # Ensure sum is exactly N/2 by small adjustment if needed?
    # The binary search should be accurate enough.
    # Let's verify sum.
    current_sum = np.sum(h)
    if abs(current_sum - N/2.0) > 1e-6:
        # Adjust linearly
        diff = N/2.0 - current_sum
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        # Re-adjust if clipping changed sum
        # For now, assume binary search is good.
        
    return _max_overlap(h)

def construct():
    """
    Construct the sequence of step heights.
    """
    # Optimize log(w)
    # w ranges from ~1 to ~100
    # log(w) ranges from 0 to ~4.6
    
    res = minimize_scalar(_objective_1d, bounds=(0.0, np.log(N)), method='bounded', options={'xatol': 1e-4})
    
    best_w_log = res.x
    best_w = np.exp(best_w_log)
    
    c = _solve_c(best_w)
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

# EVOLVE-BLOCK-START
# (The construct function is defined above, but we need to keep the interface)
# The task says "Keep the function names and signatures of the seed program."
# The seed program has construct() -> sequence of step heights.
# So we just need to make sure construct() is defined and returns the right thing.
# EVOLVE-BLOCK-END
