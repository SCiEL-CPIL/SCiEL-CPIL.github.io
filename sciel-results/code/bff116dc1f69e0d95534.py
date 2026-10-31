"""Improved program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 100

def _max_overlap_fft(h):
    """
    Compute the maximum circular cross-correlation of h with 1-h.
    overlap(s) = sum_i h[i] * (1-h[i+s])
    Using FFT:
    corr = ifft(fft(h) * conj(fft(1-h)))
    """
    one_minus_h = 1.0 - h
    H = np.fft.fft(h)
    G = np.fft.fft(one_minus_h)
    corr = np.fft.ifft(H * np.conj(G)).real
    return np.max(corr)

def _construct_h_logistic(w, c):
    """
    Construct h using logistic sigmoid.
    h[i] = 1 / (1 + exp(-(i - c) / w))
    """
    i = np.arange(N)
    if w < 1e-6:
        w = 1e-6
    z = (i - c) / w
    z = np.clip(z, -50, 50)
    h = 1.0 / (1.0 + np.exp(-z))
    return h

def _solve_c_logistic(w):
    """
    Find c such that sum(h) = N/2.
    """
    target_sum = N / 2.0
    lo = -50.0
    hi = float(N) + 50.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_mid = _construct_h_logistic(w, mid)
        s_mid = np.sum(h_mid)
        if s_mid > target_sum:
            lo = mid
        else:
            hi = mid
            
    c_opt = (lo + hi) / 2.0
    return c_opt

def _get_h_for_width(w):
    """
    Get the sequence h for a given width w, ensuring sum constraint.
    """
    if w < 1e-4:
        w = 1e-4
    if w > N:
        w = N
        
    c = _solve_c_logistic(w)
    h = _construct_h_logistic(w, c)
    
    # Correct for any small numerical drift in the sum
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    if abs(diff) > 1e-9:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        for _ in range(5):
            current_sum = np.sum(h)
            diff = N / 2.0 - current_sum
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                mask = h < 1.0
                if not np.any(mask): break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                h[indices] = np.minimum(1.0, h[indices] + step)
            else:
                mask = h > 0.0
                if not np.any(mask): break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                h[indices] = np.maximum(0.0, h[indices] + step)
                
    return h

def _objective_log_w(log_w):
    """
    Objective function for optimization over log(w).
    """
    w = np.exp(log_w)
    if w < 1e-4 or w > N:
        return 1.0
    h = _get_h_for_width(w)
    ov = _max_overlap_fft(h)
    return ov

def construct():
    """
    Construct the sequence of step heights.
    """
    log_w_min = -4.6
    log_w_max = 4.6
    num_grid = 200
    
    log_w_grid = np.linspace(log_w_min, log_w_max, num_grid)
    
    best_score = float('inf')
    best_log_w = None
    best_h = None
    
    for lw in log_w_grid:
        w = np.exp(lw)
        if w < 1e-4 or w > N:
            continue
        h = _get_h_for_width(w)
        ov = _max_overlap_fft(h)
        if ov < best_score:
            best_score = ov
            best_log_w = lw
            best_h = h
            
    if best_log_w is not None:
        lower_bound = max(log_w_min, best_log_w - 0.5)
        upper_bound = min(log_w_max, best_log_w + 0.5)
        
        res = minimize_scalar(_objective_log_w, bounds=(lower_bound, upper_bound), method='bounded', options={'xatol': 1e-8})
        
        if res.fun < best_score:
            best_score = res.fun
            best_log_w = res.x
            best_h = _get_h_for_width(np.exp(best_log_w))
            
    if best_h is None:
        return np.full(N, 0.5)
        
    return best_h
