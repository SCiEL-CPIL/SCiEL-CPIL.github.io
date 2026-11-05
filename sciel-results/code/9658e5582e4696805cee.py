import numpy as np

N = 400

def construct():
    # New Approach: Random Restart Hill Climbing from Random Initializations
    # instead of perturbing the constant seed.
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # We will try several random starting points
    for restart in range(20):
        # Initialize with random normal
        h = np.random.randn(N)
        # Normalize sum to 1 to fix scale (optional, as V is scale invariant)
        # But keeping magnitude reasonable helps numerical stability
        h = h / np.std(h) if np.std(h) > 0 else h
        
        current_val = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
        
        # Simple hill climbing
        for step in range(100):
            # Generate small perturbation
            delta = np.random.randn(N) * 0.1
            new_h = h + delta
            
            new_sum = np.sum(new_h)
            if new_sum == 0:
                continue
            
            new_val = abs(2 * N * np.max(np.convolve(new_h, new_h)) / new_sum ** 2)
            
            if new_val < current_val:
                h = new_h
                current_val = new_val
                
        if current_val < best_val:
            best_val = current_val
            best_h = h
            
    return best_h
