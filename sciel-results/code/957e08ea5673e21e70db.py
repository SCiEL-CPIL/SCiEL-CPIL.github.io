"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import random


def construct():
    # Generate Type 1 vectors (norm squared 4)
    type1_vectors = []
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            type1_vectors.append(v)
    
    # Generate Type 3 vectors (norm squared 4)
    type3_vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for l in range(k+1, D):
                    for signs in product((1, -1), repeat=4):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j], v[k], v[l] = signs
                        type3_vectors.append(v)
    
    # Type 1 vectors are compatible with all Type 3 vectors (dot product <= 2)
    # and with each other.
    # So we can always include all Type 1 vectors.
    selected = list(type1_vectors)
    
    # We need to select a subset of Type 3 vectors such that pairwise dot product <= 2.
    # Greedy with random restarts to find a larger independent set.
    
    best_selected_type3 = []
    
    # Run multiple random greedy passes
    for _ in range(50):
        random.shuffle(type3_vectors)
        current_selected = []
        for v in type3_vectors:
            ok = True
            for w in current_selected:
                if np.dot(v, w) > 2:
                    ok = False
                    break
            if ok:
                current_selected.append(v)
        
        if len(current_selected) > len(best_selected_type3):
            best_selected_type3 = current_selected
    
    # Combine Type 1 and best Type 3
    final_selected = selected + best_selected_type3
    
    if len(final_selected) > 0:
        return np.array(final_selected, dtype=np.int64)
    else:
        # Fallback
        return np.array(type1_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
