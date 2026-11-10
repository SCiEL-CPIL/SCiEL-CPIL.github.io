import numpy as np
from itertools import combinations
import time

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580  # safety margin
    
    # Generate all candidate vectors with exactly 4 non-zero coords (±1), norm² = 4
    candidates = []
    for combo in combinations(range(d), 4):
        for signs in range(16):
            p = np.zeros(d, dtype=np.int64)
            for i, idx in enumerate(combo):
                p[idx] = 1 if (signs >> i) & 1 else -1
            candidates.append(p)
    
    n_cand = len(candidates)
    N = 4  # norm squared
    
    # Precompute dot products matrix (or compute on the fly)
    # For greedy: pick a candidate, then all others must have dot <= N/2 = 2
    
    # Use greedy with randomization and multiple trials
    best_set = []
    best_size = 0
    
    np.random.seed(42)
    
    # Precompute all dot products as a matrix for speed
    # 5280 x 5280 matrix of ints
    cand_arr = np.array(candidates, dtype=np.int64)
    dots = cand_arr @ cand_arr.T  # 5280 x 5280
    
    max_dot_allowed = N // 2  # = 2
    
    # Greedy: repeatedly pick a vertex with minimum degree in remaining graph
    # where edges connect pairs with dot > max_dot_allowed
    
    # Build adjacency: edge between i,j if dots[i,j] > max_dot_allowed
    # We want a maximum independent set in this graph
    
    # Greedy approach: maintain a set of "forbidden" indices
    # Start with all candidates available
    
    n_trials = 50
    for trial in range(n_trials):
        if time.time() > deadline - 30:
            break
        
        # Random ordering
        order = np.random.permutation(n_cand)
        
        selected = []
        forbidden = np.zeros(n_cand, dtype=bool)
        
        for idx in order:
            if forbidden[idx]:
                continue
            # Check if this candidate is compatible with all already selected
            # (it should be, since we mark forbidden when we select)
            selected.append(idx)
            # Mark all incompatible candidates as forbidden
            incompatible = dots[idx] > max_dot_allowed
            forbidden |= incompatible
        
        size = len(selected)
        if size > best_size:
            best_size = size
            best_set = selected[:]
        
        # Local improvement: try to add more or swap
        # For now, just do greedy
    
    if best_size == 0:
        # Fallback: return the D11 lattice
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = s1
                        p[j] = s2
                        points.append(p)
        return np.array(points, dtype=np.int64)
    
    result = cand_arr[best_set]
    return np.around(result).astype(np.int64)
