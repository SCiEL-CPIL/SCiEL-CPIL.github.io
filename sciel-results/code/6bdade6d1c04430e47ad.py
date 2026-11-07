import numpy as np

N = 100

def construct():
    # Start with random binary sequence with 50 ones
    h = np.zeros(N)
    h[:50] = 1.0
    
    # Shuffle
    np.random.shuffle(h)
    
    def calc_value(h):
        # Compute max correlation
        # C_k = sum h_i (1-h_{i+k})
        # Use correlate
        # 1-h is the complement
        g = 1 - h
        corr = np.correlate(h, g, mode="full")
        return np.max(corr) / N * 2

    best_h = h.copy()
    best_val = calc_value(best_h)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    # Precompute shifts for speed?
    # N=100 is small.
    
    import time
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    while T > T_min and time.time() < deadline:
        # Pick two indices to swap
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i == j or h[i] == h[j]:
            continue
            
        # Swap
        h[i], h[j] = h[j], h[i]
        
        new_val = calc_value(h)
        
        if new_val < best_val or np.random.rand() < np.exp((best_val - new_val) / T):
            best_h = h.copy()
            best_val = new_val
        else:
            # Revert
            h[i], h[j] = h[j], h[i]
            
        T *= alpha
        
    return best_h
