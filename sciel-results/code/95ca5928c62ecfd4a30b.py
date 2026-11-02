"""Erdos minimum overlap: binary pattern optimization via simulated annealing."""
import numpy as np
import time

N = 100
HALF = N // 2

def _compute_max_overlap(h):
    """Compute max overlap of h with shifted 1-h for binary h with HALF ones."""
    max_overlap = 0.0
    for k in range(1, N):
        c_k = np.sum(h * np.roll(h, -k))
        overlap_k = HALF - c_k
        if overlap_k > max_overlap:
            max_overlap = overlap_k
    return max_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    np.random.seed(42)
    
    # Initial random pattern
    h = np.zeros(N, dtype=np.int8)
    indices = np.random.permutation(N)
    h[indices[:HALF]] = 1
    
    h_score = _compute_max_overlap(h)
    best_score = h_score
    best_h = h.copy()
    
    # Simulated annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    step = 0
    while T > T_min:
        if time.time() > deadline:
            break
        
        # Pick a random position to flip
        pos = np.random.randint(N)
        old_val = h[pos]
        
        if old_val == 1:
            zero_positions = np.where(h == 0)[0]
            if len(zero_positions) == 0:
                T *= cooling
                continue
            pos2 = zero_positions[np.random.randint(len(zero_positions))]
            
            h[pos] = 0
            h[pos2] = 1
            new_score = _compute_max_overlap(h)
            
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
            
            if new_score < h_score or np.random.random() < np.exp(-(new_score - h_score) / T):
                h_score = new_score
            else:
                h[pos] = 1
                h[pos2] = 0
        else:
            one_positions = np.where(h == 1)[0]
            if len(one_positions) == 0:
                T *= cooling
                continue
            pos2 = one_positions[np.random.randint(len(one_positions))]
            
            h[pos] = 1
            h[pos2] = 0
            new_score = _compute_max_overlap(h)
            
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
            
            if new_score < h_score or np.random.random() < np.exp(-(new_score - h_score) / T):
                h_score = new_score
            else:
                h[pos] = 0
                h[pos2] = 1
        
        T *= cooling
        step += 1
    
    # Return as float array
    return best_h.astype(np.float64)
