"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import numpy as np
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 115

    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    # Generate all candidates with norm squared 8
    candidates = []
    
    # Type 1: (±2, ±2, 0, ...)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                candidates.append(v)
                
    # Type 2: (±1, ..., ±1, 0, ...) with 8 ones
    for pos_set in itertools.combinations(range(D), 8):
        for mask in range(256):
            v = [0] * D
            for idx, pos in enumerate(pos_set):
                if mask & (1 << idx):
                    v[pos] = -1
                else:
                    v[pos] = 1
            candidates.append(v)
            
    # Type 3: (±2, ±1, ..., ±1, 0, ...) with one 2 and four 1s
    for pos_2 in range(D):
        others = [p for p in range(D) if p != pos_2]
        for pos_set_1 in itertools.combinations(others, 4):
            for s2 in (1, -1):
                for mask in range(16):
                    v = [0] * D
                    v[pos_2] = 2 * s2
                    for idx, pos in enumerate(pos_set_1):
                        v[pos] = -1 if (mask & (1 << idx)) else 1
                    candidates.append(v)

    candidates_arr = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates_arr)
    
    best_arr = np.array([], dtype=np.int64).reshape(0, D)
    
    # Try multiple random orders to find a better packing
    random.seed(42)
    
    # We'll try a few restarts if time permits
    for restart in range(10):
        if not elapsed_ok():
            break
            
        indices = np.arange(n_candidates)
        random.shuffle(indices)
        
        selected_indices = []
        current_arr = np.array([], dtype=np.int64).reshape(0, D)
        
        # Process candidates in the shuffled order
        for idx in indices:
            if not elapsed_ok():
                break
                
            v = candidates_arr[idx]
            
            if len(current_arr) == 0:
                current_arr = np.array([v], dtype=np.int64)
                selected_indices.append(idx)
            else:
                # Check if v is compatible with all in current_arr
                # Condition: max(dot(current_arr, v)) <= 4
                dots = current_arr @ v
                if np.max(dots) <= 4:
                    current_arr = np.vstack([current_arr, v])
                    selected_indices.append(idx)
                    
        if len(current_arr) > len(best_arr):
            best_arr = current_arr.copy()
            
            # If we've found a very good solution, we can stop early
            if len(current_arr) >= 450:
                break

    return best_arr

# EVOLVE-BLOCK-END
