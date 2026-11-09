"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Initialize with a random binary sequence with exactly N/2 ones
    # This ensures sum is N/2 = 50
    h = np.zeros(N, dtype=np.float64)
    indices = np.random.choice(N, N // 2, replace=False)
    h[indices] = 1.0
    
    # Function to compute the score: max overlap of h with shifted (1-h)
    def get_score(h):
        g = 1.0 - h
        # Cross correlation: sum h[i] * g[i+k]
        # np.correlate(h, g, mode='full') computes this for all shifts
        corr = np.correlate(h, g, mode='full')
        return np.max(corr)
    
    current_score = get_score(h)
    best_h = h.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 1.0
    T_min = 0.01
    alpha = 0.9999  # Cooling rate
    
    while time.time() < deadline:
        # Pick two random indices to swap
        # Using randint is faster than choice with replace=False
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        
        # Skip if same index or same value (swap does nothing)
        if i == j or h[i] == h[j]:
            continue
            
        # Perform swap
        h[i], h[j] = h[j], h[i]
        
        new_score = get_score(h)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_score - current_score
        if delta < 0 or (T > 0 and np.random.rand() < np.exp(-delta / T)):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            # Revert swap
            h[i], h[j] = h[j], h[i]
            
        T *= alpha
        if T < T_min:
            T = T_min

    return best_h
# EVOLVE-BLOCK-END
