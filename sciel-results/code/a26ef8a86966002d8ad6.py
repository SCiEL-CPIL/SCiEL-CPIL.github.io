"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
import random

D = 11

# EVOLVE-BLOCK-START

def generate_candidates():
    """Generate weight-2 and weight-4 integer vectors."""
    candidates = []
    
    # Weight 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(np.array(v, dtype=np.int64))
                
    # Weight 4
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(np.array(v, dtype=np.int64))
            
    return candidates

def is_compatible(u, v):
    """Check if u and v satisfy the distance constraint."""
    # Condition: ||u-v|| >= max(||u||, ||v||)
    # Equivalent to: 2(u.v)^2 <= ||u||^2 ||v||^2
    dot = np.dot(u, v)
    norm_u_sq = np.dot(u, u)
    norm_v_sq = np.dot(v, v)
    
    # To avoid floating point issues, use integers
    # 2 * dot^2 <= norm_u_sq * norm_v_sq
    return 2 * (dot * dot) <= norm_u_sq * norm_v_sq

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    candidates = generate_candidates()
    n_cands = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.array([np.dot(c, c) for c in candidates], dtype=np.int64)
    
    # We will use a greedy approach with randomization and local search.
    # Best set found so far
    best_set = []
    best_score = 0
    
    # To speed up conflict checking, we can't easily precompute all pairs (6000x6000 is 36M, might be slow in Python).
    # We will check on the fly.
    
    # Strategy 1: Greedy with multiple restarts
    # Sort candidates by "density" or just randomize
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Randomize order
        indices = list(range(n_cands))
        random.shuffle(indices)
        
        current_set = []
        current_indices = set()
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            v = candidates[idx]
            # Check compatibility with all in current_set
            compatible = True
            for c_idx in current_indices:
                if not is_compatible(v, candidates[c_idx]):
                    compatible = False
                    break
            
            if compatible:
                current_set.append(v)
                current_indices.add(idx)
                
        if len(current_set) > best_score:
            best_score = len(current_set)
            best_set = list(current_set)
            
        # Local Search: Try to swap out one vector for another
        # This is expensive, so we only do it for the best few sets or if time permits
        if best_score > 220:
             # Try to improve best_set
             pass
             
    return np.array(best_set, dtype=np.int64) if best_set else np.array([], dtype=np.int64)

# EVOLVE-BLOCK-END
