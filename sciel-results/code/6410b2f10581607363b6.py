"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 200 # Increased N for better approximation

def _max_overlap_fft(h):
    """
    Compute the maximum overlap of h with shifted 1-h over all shifts using FFT.
    Overlap for shift s is sum(h[i] * (1-h)[i-s]).
    """
    g = 1.0 - h
    # Zero pad to 2*N to avoid circular convolution artifacts if we were doing linear, 
    # but we want circular correlation for the periodic domain [0, N).
    # Actually, the problem is on [0, 2] mapped to N steps, so it's periodic.
    # Circular correlation is correct.
    
    fft_h = np.fft.fft(h)
    fft_g = np.fft.fft(g)
    
    # Cross-correlation: sum h[i] g[i-s]
    # Ifft(fft(h) * conj(fft(g))) gives sum h[i] g[i-k]
    corr = np.fft.ifft(fft_h * np.conj(fft_g)).real
    
    # The shifts s=0..N-1 correspond to indices 0..N-1 in the correlation array
    # (assuming standard DFT ordering)
    # Let's verify:
    # k=0: sum h[i] g[i]
    # k=1: sum h[i] g[i-1]
    # ...
    # We want max over all shifts.
    return np.max(corr)

def _construct_h(w, c):
    """
    Construct h given width w and center c.
    h[i] = 0.5 + 0.5 * tanh((i - c) / w)
    """
    i = np.arange(N)
    if w < 1e-6:
        w = 1e-6
    h = 0.5 + 0.5 * np.tanh((i - c) / w)
    return h

def _solve_c(w):
    """
    Find c such that sum(h) = N/2.
    """
    target_sum = N / 2.0
    
    def sum_diff(c):
        h = _construct_h(w, c)
        return np.sum(h) - target_sum
        
    lo = 0.0
    hi = float(N)
    
    # sum(h) is decreasing with c (step moves right, more 0s, fewer 1s? 
    # No, h=0.5+0.5tanh((i-c)/w).
    # i < c -> tanh < 0 -> h < 0.5.
    # i > c -> tanh > 0 -> h > 0.5.
    # As c increases, the boundary moves right.
    # More i's are < c (h~0) and fewer are > c (h~1)?
    # Domain is 0..N-1.
    # If c=0, most i > 0, so h~1. Sum ~ N.
    # If c=N, most i < N, so h~0. Sum ~ 0.
    # So sum is decreasing with c.
    
    for _ in range(60):
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
    if w < 1e-3:
        w = 1e-3
    if w > N:
        w = N
        
    c = _solve_c(w)
    h = _construct_h(w, c)
    
    # Enforce exact sum
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    
    if abs(diff) > 1e-9:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        
        # Iterative fix for exact sum
        for _ in range(50):
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

    return _max_overlap_fft(h)

def construct():
    """
    Construct the sequence of step heights.
    """
    # Optimize log(w)
    # w ranges from ~1 to ~N
    # log(w) ranges from 0 to log(N)
    
    # Grid search first to find a good region
    w_logs = np.linspace(0.0, np.log(N), 50)
    best_val = np.inf
    best_w_log = w_logs[0]
    
    for wl in w_logs:
        val = _objective_1d(wl)
        if val < best_val:
            best_val = val
            best_w_log = wl
            
    # Local refinement
    res = minimize_scalar(_objective_1d, bounds=(best_w_log - 0.5, best_w_log + 0.5), method='bounded', options={'xatol': 1e-6})
    
    best_w_log = res.x
    best_w = np.exp(best_w_log)
    
    c = _solve_c(best_w)
    h = _construct_h(best_w, c)
    
    # Final correction for exact sum
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    
    if abs(diff) > 1e-10:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        
        for _ in range(100):
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
