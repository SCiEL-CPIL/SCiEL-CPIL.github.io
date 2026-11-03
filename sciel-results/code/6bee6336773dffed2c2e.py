"""Minimize normalized max autoconvolution via optimized smooth peak shapes."""
import numpy as np
import time

N = 600
DEADLINE = 120.0  # wall-clock seconds; safety margin below

def _autoconv_max_ratio(heights):
    """Compute max(autoconv) / ||h||_2^2 for a 1D step function."""
    h = np.asarray(heights, dtype=np.float64)
    if np.any(h < 0):
        return np.inf
    norm_sq = np.dot(h, h)
    if norm_sq <= 0:
        return np.inf
    # Linear autoconvolution (full)
    ac = np.convolve(h, h, mode='full')
    return float(np.max(ac) / norm_sq)

def _gaussian_heights(width_param, center=0.0, n=N):
    """Generate Gaussian-like heights on n equal intervals in [-1/4, 1/4]."""
    # Intervals are centered at positions
    # Interval i covers [x_i, x_i+dx] where dx = (1/2)/n
    dx = 0.5 / n
    # Centers of intervals
    x = np.linspace(-0.25 + dx/2, 0.25 - dx/2, n)
    # width_param controls spread; use sigma = width_param * dx * 10
    # Actually let's think in terms of number of intervals
    sigma_intervals = max(1.0, width_param)
    sigma_x = sigma_intervals * dx
    h = np.exp(-0.5 * ((x - center) / sigma_x) ** 2)
    return h

def _raised_cosine_heights(width_intervals, center=0.0, n=N):
    """Raised cosine bump: h(x) = 0.5*(1+cos(pi*x/w)) for |x|<=w, else 0."""
    dx = 0.5 / n
    x = np.linspace(-0.25 + dx/2, 0.25 - dx/2, n)
    w = width_intervals * dx
    h = np.zeros(n)
    mask = np.abs(x - center) <= w
    h[mask] = 0.5 * (1 + np.cos(np.pi * (x[mask] - center) / w))
    return h

def construct():
    start_time = time.time()
    
    # Initial best: flat baseline
    best_heights = [1.0] * N
    best_score = _autoconv_max_ratio(best_heights)
    
    # Try a range of Gaussian widths
    # width_param in terms of number of intervals (sigma)
    # Small sigma = narrow peak
    candidate_widths = np.logspace(-0.5, 2.0, 80)  # from ~0.3 to ~100 intervals
    
    for wp in candidate_widths:
        if time.time() - start_time > DEADLINE - 5:
            break
        h = _gaussian_heights(wp)
        s = _autoconv_max_ratio(h)
        if s < best_score:
            best_score = s
            best_heights = h.tolist()
    
    # Try raised cosine widths
    rc_widths = np.linspace(2, 50, 50)
    for wp in rc_widths:
        if time.time() - start_time > DEADLINE - 5:
            break
        h = _raised_cosine_heights(wp)
        s = _autoconv_max_ratio(h)
        if s < best_score:
            best_score = s
            best_heights = h.tolist()
    
    # Local refinement around best Gaussian width
    if time.time() - start_time < DEADLINE - 10:
        # Find which width was best among Gaussians
        # Re-evaluate and narrow search
        refined_widths = np.logspace(-1.0, 1.5, 100)
        for wp in refined_widths:
            if time.time() - start_time > DEADLINE - 3:
                break
            h = _gaussian_heights(wp)
            s = _autoconv_max_ratio(h)
            if s < best_score:
                best_score = s
                best_heights = h.tolist()
    
    # Ensure nonnegative
    best_heights = [max(0.0, v) for v in best_heights]
    
    return best_heights
