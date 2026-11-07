"""Improved program using alternating pattern and multi-start optimization."""
import numpy as np
from scipy.optimize import minimize

N = 100

def _objective(x):
    h = (x + 1.0) / 2.0
    # Correlation with 1-h
    corr = np.correlate(h, 1.0 - h, mode="full")
    val = np.max(corr) / N * 2.0
    # Penalty for sum constraint (should be 0)
    sum_penalty = np.abs(np.sum(h) - N / 2.0) * 1000.0
    # Penalty for out-of-bounds (shouldn't happen due to bounds, but just in case)
    bound_penalty = np.sum(np.maximum(0, h - 1.0)) + np.sum(np.maximum(0, -h))
    return val + sum_penalty + bound_penalty

def _fix_sum(h):
    """Ensure the sum is exactly N/2 by distributing the error."""
    current_sum = np.sum(h)
    target_sum = N / 2.0
    diff = current_sum - target_sum
    
    if abs(diff) > 1e-12:
        # Distribute the correction evenly to maintain the pattern
        h = h - diff / N
        # Clip and redistribute if needed
        h = np.clip(h, 0.0, 1.0)
        current_sum = np.sum(h)
        diff = current_sum - target_sum
        if abs(diff) > 1e-12:
            # If still off, adjust the elements with most room
            room_to_1 = 1.0 - h
            room_to_0 = h
            if diff > 0:
                # Need to decrease, adjust elements with most room to 0
                idx = np.argsort(room_to_0)[:10]
                per_adjust = diff / len(idx)
                h[idx] -= per_adjust
            else:
                # Need to increase, adjust elements with most room to 1
                idx = np.argsort(room_to_1)[:10]
                per_adjust = -diff / len(idx)
                h[idx] += per_adjust
            h = np.clip(h, 0.0, 1.0)
    
    return h

def construct():
    best_h = np.full(N, 0.5)
    best_val = np.max(np.correlate(best_h, 1.0 - best_h, mode="full")) / N * 2.0
    
    bounds = [(-1.0, 1.0)] * N
    
    # Try multiple initializations with different frequencies
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        # Create alternating pattern: high, low, high, low, ...
        # Use sinusoidal with the given frequency
        t = np.linspace(0, freq * 2 * np.pi, N)
        # Alternate between high and low
        x0 = 0.8 * np.sin(t)
        
        res = minimize(
            _objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        h_opt = (res.x + 1.0) / 2.0
        h_opt = _fix_sum(h_opt)
        
        val_opt = np.max(np.correlate(h_opt, 1.0 - h_opt, mode="full")) / N * 2.0
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    # Also try a simple alternating pattern
    alt_h = np.where(np.arange(N) % 2 == 0, 0.8, 0.2)
    alt_h = _fix_sum(alt_h)
    val_alt = np.max(np.correlate(alt_h, 1.0 - alt_h, mode="full")) / N * 2.0
    if val_alt < best_val:
        best_val = val_alt
        best_h = alt_h
    
    # Try another alternating pattern with different values
    alt_h2 = np.where(np.arange(N) % 2 == 0, 0.7, 0.3)
    alt_h2 = _fix_sum(alt_h2)
    val_alt2 = np.max(np.correlate(alt_h2, 1.0 - alt_h2, mode="full")) / N * 2.0
    if val_alt2 < best_val:
        best_val = val_alt2
        best_h = alt_h2
    
    return best_h
