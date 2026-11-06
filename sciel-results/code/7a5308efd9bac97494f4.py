import numpy as np
import time

N = 400

def _value(h):
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2 * N * max_conv / (s ** 2))

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    # Try a few random perturbations with alternating structure
    rng = np.random.default_rng(42)
    
    for i in range(50):
        if time.time() > deadline:
            break
        
        # Generate a candidate: mostly 1s with some alternating noise
        # Or just random phases?
        # Let's try to flatten the convolution.
        
        # Start from best_h or random
        if i % 5 == 0:
            h = np.ones(N)
            # Add alternating noise
            noise = rng.normal(0, 0.1, N) * np.array([1, -1] * (N // 2))
            h = h + noise
        else:
            h = best_h + rng.normal(0, 0.01, N)
            
        # Normalize sum to 1 to keep scale invariant?
        # Value is scale invariant, so no need.
        
        val = _value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    return best_h
