import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initialize random binary sequence
    x = np.random.choice([-1, 1], size=N)
    
    # Compute initial autocorrelation
    c = compute_ac(x)
    max_val = np.max(np.abs(c))
    
    best_x = x.copy()
    best_max = max_val
    
    while time.time() < deadline:
        # Pick random index to flip
        i = np.random.randint(N)
        
        # Compute change in autocorrelation
        # c[k] changes by -2 * x[i] * (x[i+k] + x[i-k])
        # We update c in place
        
        # Save old c? No, we can just revert if needed.
        # But we need to know which c[k] changed to revert.
        # It's easier to just compute new c from scratch if N is small?
        # N=400. FFT is fast. But O(N) update is faster.
        # Let's try O(N) update.
        
        old_x_i = x[i]
        x[i] = -old_x_i
        
        # Update c
        # For k in 0..N-1-i: c[k] -= 2 * old_x_i * x[i+k]  (Note: x[i] is flipped, so delta is -2 * old_x_i * x[i+k])
        # Wait. c[k] = sum x[j]x[j+k].
        # Old term: old_x_i * x[i+k]. New term: -old_x_i * x[i+k].
        # Change: -2 * old_x_i * x[i+k].
        
        # Similarly for k where i-k >= 0: c[k] -= 2 * old_x_i * x[i-k]
        
        # We can vectorize this update.
        # Indices for positive lags: k = 0, ..., N-1-i. j = i+k.
        # Indices for negative lags: k = 1, ..., i. j = i-k.
        
        # Let's just recompute c using FFT if it's simpler and fast enough.
        # FFT of 800 points is ~10us.
        # 100,000 iterations * 10us = 1s. This is very fast.
        # So we can just recompute c every time.
        
        c = compute_ac(x)
        new_max = np.max(np.abs(c))
        
        if new_max < best_max:
            best_max = new_max
            best_x = x.copy()
            # Accept move
        else:
            # Revert
            x[i] = old_x_i
            # c is not used for decision if we revert, but we need to restore state for next iteration?
            # We recompute c every time, so we don't need to restore c.
            
    return best_x

def compute_ac(x):
    # Linear autocorrelation
    n = len(x)
    size = 2 * n
    X = np.fft.fft(x, n=size)
    ac = np.fft.ifft(X * np.conj(X)).real
    return ac[:n]
