"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, 20s margin
    
    # Initial random binary sequence with N/2 ones
    h = np.zeros(N)
    h[:N//2] = 1.0
    
    # Shuffle
    np.random.shuffle(h)
    
    # Function to calculate max overlap
    def max_overlap(h):
        max_ov = 0.0
        for k in range(N):
            # Linear shift overlap
            # O_k = sum h[i] * (1 - h[i-k]) for valid i
            # i must be in [0, N-1] and i-k in [0, N-1]
            # So i in [k, N-1]
            if k == 0:
                # O_0 = sum h[i] * (1 - h[i]) = 0 for binary
                ov = 0.0
            else:
                # h[k:] and 1-h[:-k]
                ov = np.sum(h[k:] * (1.0 - h[:-k]))
            if ov > max_ov:
                max_ov = ov
        return max_ov

    current_h = h.copy()
    current_score = max_overlap(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    # Local Search / Simulated Annealing
    # We want to minimize max_overlap
    # Swap a 1 and a 0
    
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick random indices to swap
        i = np.random.randint(N)
        j = np.random.randint(N)
        
        if i == j or current_h[i] == current_h[j]:
            continue
            
        # Swap
        current_h[i], current_h[j] = current_h[j], current_h[i]
        
        new_score = max_overlap(current_h)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        else:
            # Revert
            current_h[i], current_h[j] = current_h[j], current_h[i]
            
        T *= alpha
        
    return best_h
# EVOLVE-BLOCK-END
