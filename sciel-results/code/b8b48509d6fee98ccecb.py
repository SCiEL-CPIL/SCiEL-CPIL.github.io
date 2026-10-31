"""Constructing a kissing configuration in 11 dimensions using a greedy selection of norm-2 vectors."""

import itertools
import numpy as np

D = 11

def construct():
    # Generate all integer vectors with norm squared 4
    # Type A: One entry is +/-2, rest 0
    # Type B: Four entries are +/-1, rest 0
    
    vectors = []
    
    # Add Type A vectors (always valid)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    # Generate Type B candidates
    candidates = []
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, idx in enumerate(idxs):
                v[idx] = signs[k]
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    
    selected_b = []
    
    # Greedy selection
    # We check if the new vector is compatible with all previously selected Type B vectors.
    # Condition: dot product <= 2
    
    for v in candidates:
        if not selected_b:
            selected_b.append(v)
            continue
            
        # Convert current selected list to array for vectorized dot product
        # Note: This conversion happens every iteration. 
        # Optimization: We could maintain the array, but appending is O(N).
        # Given N is small (<5000) and D=11, this is acceptable.
        sel_arr = np.array(selected_b)
        dots = sel_arr @ v
        
        if np.all(dots <= 2):
            selected_b.append(v)
            
    # Add selected Type B vectors
    for v in selected_b:
        vectors.append(v.tolist())
        
    return np.array(vectors, dtype=np.int64)
