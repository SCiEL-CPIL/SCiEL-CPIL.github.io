import numpy as np

N = 400

def construct():
    best_h = np.ones(N)
    best_val = 2.0
    
    # Simple random search
    for _ in range(50):
        # Generate random perturbation
        x = np.random.randn(N)
        x -= np.mean(x) # Zero mean to keep sum close to N
        
        # Scale perturbation
        # We want to try different scales
        scale = np.random.uniform(0, 1)
        h = best_h + scale * x
        
        # Ensure sum is not zero
        if np.sum(h) == 0:
            continue
            
        val = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
        
        if val < best_val:
            best_val = val
            best_h = h
            
    return best_h
