import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, 10s safety
    
    # Initial guess: Random +/- 1
    x = np.random.choice([-1, 1], size=N)
    
    # Calculate initial score
    # We assume denominator is (sum(abs(x)))^2
    # For +/- 1, sum(abs(x)) = N, so denom = N^2
    # We just need to minimize max(abs(autoconv))
    
    best_x = x.copy()
    best_max_conv = np.max(np.abs(np.convolve(x, x, 'full')))
    
    # Hill Climbing
    # To make it faster, we can try to update max_conv efficiently
    # But np.convolve is C-optimized.
    # Let's see how many iterations we can do.
    # 1 convolve ~ 0.1 ms?
    # 110s / 0.1ms = 1,100,000 iterations.
    
    current_max_conv = best_max_conv
    x = best_x.copy()
    
    # Precompute indices for potential speedup?
    # No, let's just run the loop.
    
    while time.time() < deadline:
        # Pick random index
        i = np.random.randint(N)
        
        # Flip
        x[i] *= -1
        
        # Calculate new max conv
        # Use FFT for speed?
        # N=400, FFT size 800.
        # np.fft.fft is fast.
        # But np.convolve might be faster for small N due to overhead?
        # Let's stick to np.convolve for simplicity and correctness.
        
        c = np.convolve(x, x, 'full')
        max_c = np.max(np.abs(c))
        
        if max_c < current_max_conv:
            current_max_conv = max_c
            best_x = x.copy()
        else:
            # Revert
            x[i] *= -1
            
    return best_x
