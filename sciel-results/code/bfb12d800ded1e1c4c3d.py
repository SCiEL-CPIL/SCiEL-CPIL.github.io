"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # Generate all weight-4 vectors with entries in {-1, 1}
    # Support: choose 4 positions out of 11, assign ±1 to each
    candidates = []
    supports = list(itertools.combinations(range(D), 4))
    for sup in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(sup, signs):
                v[idx] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute dot product matrix (n_cand x n_cand)
    # Two vectors are compatible if dot <= 2
    # For weight-4 ±1 vectors, ||v||^2 = 4, so ||v-w||^2 = 8 - 2*v·w
    # Need ||v-w|| >= 2 (max norm), so 8 - 2*v·w >= 4 => v·w <= 2
    
    # Greedy selection: pick vectors in order, maintain compatibility
    # For speed, process in batches and use vectorized dot products
    
    selected = []
    selected_set = set()
    
    # Precompute: for each candidate, check against all selected
    # Use a greedy approach with periodic batch processing
    
    # First pass: try all candidates in a smart order
    # Sort by some heuristic (e.g., random shuffle for diversity)
    indices = np.random.permutation(n_cand)
    
    for idx in indices:
        if time.time() > deadline:
            break
        
        v = candidates[idx]
        
        if len(selected) == 0:
            selected.append(v)
            selected_set.add(idx)
            continue
        
        # Check compatibility with all selected vectors
        # dot = selected_vectors @ v
        # Need all dots <= 2
        sel_arr = np.array(selected, dtype=np.int64)
        dots = sel_arr @ v
        
        if np.all(dots <= 2):
            selected.append(v)
            selected_set.add(idx)
    
    if len(selected) == 0:
        # Fallback: return seed solution
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
