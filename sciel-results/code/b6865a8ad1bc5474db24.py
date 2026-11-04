import numpy as np
import time

N = 600
DELTA = 1.0 / 1200.0

def compute_autoconv_max(h, num_t=201):
    h = np.asarray(h, dtype=float)
    total = np.sum(h)
    if total <= 0:
        return float('inf')
    h = h * (1.0 / (total * DELTA))
    
    a = -0.25 + np.arange(N) * DELTA
    a_col = a[:, None]
    a_row = a[None, :]
    h_col = h[:, None]
    h_row = h[None, :]
    
    t_vals = np.linspace(-0.5, 0.5, num_t)
    max_val = 0.0
    for t in t_vals:
        lb = np.maximum(a_col, t - a_row - DELTA)
        rb = np.minimum(a_col + DELTA, t - a_row)
        overlap = np.maximum(0.0, rb - lb)
        val = np.sum(h_col * h_row * overlap)
        if val > max_val:
            max_val = val
    return max_val

def construct():
    start = time.time()
    deadline = start + 130.0
    
    x = -0.25 + (np.arange(N) + 0.5) * DELTA
    
    candidates = [
        np.ones(N),
        np.maximum(np.cos(np.pi * x / 0.25), 0),
        np.exp(-20 * x**2),
        np.maximum(1 - (x/0.25)**2, 0),
        np.maximum(np.sin(np.pi * x / 0.25), 0),
    ]
    
    best_h = candidates[0]
    best_val = compute_autoconv_max(best_h)
    
    for hc in candidates[1:]:
        if time.time() > deadline:
            break
        if np.sum(hc) <= 0:
            continue
        v = compute_autoconv_max(hc)
        if v < best_val:
            best_val, best_h = v, hc
    
    return list(best_h)
