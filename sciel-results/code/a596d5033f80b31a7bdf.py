"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import random

def construct():
    # Generate candidate vectors with squared norm = 4
    candidates = []
    
    # Type 1: one entry is ±2, rest 0 (Norm^2 = 4)
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    
    # Type 3: four entries are ±1, rest 0 (Norm^2 = 4)
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for l in range(k+1, D):
                    for signs in product((1, -1), repeat=4):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j], v[k], v[l] = signs
                        candidates.append(v)
    
    # Convert to list of lists for easier manipulation if needed, 
    # but we will work with numpy arrays in the loop for dot products.
    # To speed up dot product checks, we can pre-compute or use vectorization.
    # However, the greedy step is sequential.
    
    best_set = []
    num_restarts = 200
    
    for _ in range(num_restarts):
        # Shuffle candidates to break symmetry and explore different local optima
        idx = list(range(len(candidates)))
        random.shuffle(idx)
        
        current_set = []
        current_vectors = [] # Store numpy arrays for current set
        
        for i in idx:
            v = candidates[i]
            ok = True
            # Check against all vectors in current_set
            # Optimization: If len(current_set) is large, this is slow.
            # We can vectorize the check: dot(current_set, v)
            if len(current_vectors) > 0:
                dots = np.dot(np.array(current_vectors), v)
                if np.any(dots > 2):
                    ok = False
            if ok:
                current_set.append(v)
                current_vectors.append(v)
                
        if len(current_set) > len(best_set):
            best_set = current_set
            # Early stop if we hit a known high bound? 
            # Theoretical max for norm 4 in 11D is not trivial, but let's just run.
            
    if len(best_set) > 0:
        return np.array(best_set, dtype=np.int64)
    else:
        # Fallback
        return np.zeros((0, D), dtype=np.int64)
# EVOLVE-BLOCK-END
