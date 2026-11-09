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
                        if mask & (1 << idx):
                            v[pos] -1 if (mask & (1 << idx)) else 1 # Wait, syntax error in thought, fix below
                    # Fixing logic:
                    v = [0] * D
                    v[pos_2] = 2 * s2
                    for idx, pos in enumerate(pos_set_1):
                        v[pos] = -1 if (mask & (1 << idx)) else 1
                    candidates.append(v)

    candidates_arr = np.array(candidates, dtype=np.int64)
    
    best_vectors = []
    best_arr = np.array([], dtype=np.int64).reshape(0, D)
    
    # Try a few random orders to find a better packing
    # Given time constraints, we can afford 1-2 tries.
    # Let's do 1 try with a good shuffle.
    
    indices = np.arange(len(candidates_arr))
    random.seed(42)
    random.shuffle(indices)
    
    selected_indices = []
    current_arr = np.array([], dtype=np.int64).reshape(0, D)
    
    # We process in chunks to check time
    chunk_size = 5000
    
    for start_idx in range(0, len(indices), chunk_size):
        if not elapsed_ok():
            break
            
        chunk_indices = indices[start_idx : start_idx + chunk_size]
        
        for idx in chunk_indices:
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
        best_arr = current_arr
        best_vectors = [candidates_arr[i] for i in selected_indices]

    return np.array(best_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
