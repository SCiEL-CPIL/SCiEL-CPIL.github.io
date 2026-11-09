import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initialize with random signs
    x = np.random.choice([-1, 1], N)
    
    # Calculate initial max sidelobe
    def max_sidelobe(x):
        # Compute full autocorrelation
        # np.correlate(x, x, mode='full') gives lags -(N-1) to N-1
        # Center is at index N-1
        c = np.correlate(x, x, mode='full')
        # Sidelobes are all indices except the center
        center = N - 1
        # Mask out the center
        c_sidelobes = np.concatenate([c[:center], c[center+1:]])
        return np.max(np.abs(c_sidelobes))
    
    current_max = max_sidelobe(x)
    best_x = x.copy()
    best_max = current_max
    
    # Greedy local search
    # We try flipping each bit and see if it reduces the max sidelobe
    # To speed up, we can use a stochastic approach or check a subset
    
    # For N=400, one correlation is O(N^2) ~ 160k ops.
    # 140s / (160k ops) is plenty for many iterations if vectorized.
    # But np.correlate is fast.
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        # Try flipping each bit
        for i in range(N):
            if time.time() > deadline:
                break
            # Flip bit i
            old_val = x[i]
            x[i] = -old_val
            
            new_max = max_sidelobe(x)
            
            if new_max < best_max:
                best_max = new_max
                best_x = x.copy()
                improved = True
            else:
                # Revert
                x[i] = old_val
    
    return best_x
