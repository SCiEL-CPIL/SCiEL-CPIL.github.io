import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Initial guess
    # Try a few different initializations and pick the best
    # 1. All ones (score 2.0)
    # 2. Cosine
    # 3. Random
    
    best_h = np.ones(N)
    best_val = 2.0
    
    def evaluate(h):
        S = np.sum(h)
        if np.abs(S) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        max_c = np.max(c)
        val = abs(2 * N * max_c / (S ** 2))
        return val
    
    # Check seed
    val_seed = evaluate(best_h)
    
    # Optimization loop
    # We will run gradient descent from a few random starts
    
    rng = np.random.default_rng(42)
    
    for run in range(10):
        if time.time() > deadline - 10:
            break
            
        # Initialize
        if run == 0:
            h = np.ones(N)
        elif run == 1:
            h = np.cos(2 * np.pi * np.arange(N) / N)
        else:
            h = rng.standard_normal(N)
            # Ensure sum is not zero
            if np.abs(np.sum(h)) < 1e-6:
                h += 1.0
        
        # Normalize to sum 1
        h = h / np.sum(h)
        
        alpha = 50.0
        lr = 0.01
        prev_val = np.inf
        
        for step in range(5000):
            if time.time() > deadline - 5:
                break
            
            c = np.convolve(h, h)
            max_c = np.max(c)
            S = np.sum(h)
            val = abs(2 * N * max_c / (S ** 2))
            
            if val < best_val:
                best_val = val
                best_h = h.copy()
            
            # Smoothed max
            # To avoid overflow, shift c
            c_shifted = c - np.max(c)
            exp_c = np.exp(alpha * c_shifted)
            sum_exp = np.sum(exp_c)
            p = exp_c / sum_exp
            
            # Gradient of smoothed max
            # G_i = 2 * sum_j h_j p_{i+j}
            # Using correlation
            # corr = np.correlate(h, p, mode='full')
            # G = 2 * corr[N-1 : N-1-N : -1] ?
            # Let's verify indices.
            # corr[k] corresponds to lag k - (N-1)
            # We want lag -i.
            # Index in corr: N-1 - i.
            # i ranges 0..N-1.
            # Indices: N-1, N-2, ..., 0.
            corr = np.correlate(h, p, mode='full')
            G = 2.0 * corr[N-1 : N-1-N : -1]
            
            # Update h
            h_new = h - lr * G
            
            # Normalize
            S_new = np.sum(h_new)
            if np.abs(S_new) < 1e-12:
                h_new = h
                continue
            h_new = h_new / S_new
            
            # Check for NaNs
            if not np.all(np.isfinite(h_new)):
                break
            
            h = h_new
            
            # Adaptive learning rate
            # If val increased significantly, reduce lr
            if val > prev_val * 1.1:
                lr *= 0.5
            prev_val = val

    return best_h
