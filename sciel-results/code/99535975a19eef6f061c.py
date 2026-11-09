"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

D = 11
DEADLINE_MARGIN = 10  # seconds to spare before the 150s limit


def _seed_solution():
    """Return the seed solution: all ±ei ± ej for i<j."""
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
    return np.array(vectors, dtype=np.int64)


def construct():
    """Construct integer vectors in R^11 with min pairwise distance >= max norm."""
    start_time = time.time()
    
    # Generate candidate vectors with various norm patterns
    # Key insight: the constraint is min_pairwise_dist >= max_norm (of selected set)
    # So we should try to pack vectors that are mutually far apart
    
    # Strategy: generate candidates by weight patterns
    # Type 1: weight 1 vectors (±ei) - D of them
    # Type 2: weight 2 vectors (±ei ± ej) - D*(D-1) of them  
    # Type 3: weight 3 vectors (±ei ± ej ± ek) - D*(D-1)*(D-2) of them
    # etc.
    
    candidates = []
    
    # Add weight-1 vectors
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(tuple(v))
    
    # Add weight-2 vectors
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    candidates.append(tuple(v))
    
    # Add weight-3 vectors (sampled to keep manageable)
    if time.time() - start_time < DEADLINE_MARGIN:
        for i in range(D):
            for j in range(D):
                if i == j:
                    continue
                for k in range(D):
                    if k == i or k == j:
                        continue
                    for si, sj, sk in product((1, -1), repeat=3):
                        v = [0] * D
                        v[i], v[j], v[k] = si, sj, sk
                        candidates.append(tuple(v))
    
    # Add weight-4 vectors (sampled)
    if time.time() - start_time < DEADLINE_MARGIN:
        import random
        random.seed(42)
        count_w4 = 0
        max_w4 = 2000
        for i in range(D):
            for j in range(D):
                if i == j:
                    continue
                for k in range(D):
                    if k == i or k == j:
                        continue
                    for l in range(D):
                        if l == i or l == j or l == k:
                            continue
                        if count_w4 >= max_w4:
                            break
                        for si, sj, sk, sl in product((1, -1), repeat=4):
                            v = [0] * D
                            v[i], v[j], v[k], v[l] = si, sj, sk, sl
                            candidates.append(tuple(v))
                            count_w4 += 1
                        if count_w4 >= max_w4:
                            break
                    if count_w4 >= max_w4:
                        break
                if count_w4 >= max_w4:
                    break
            if count_w4 >= max_w4:
                break
    
    if not candidates:
        return _seed_solution()
    
    # Remove duplicates
    candidates = list(set(candidates))
    cand_array = np.array(candidates, dtype=np.int64)
    n_cand = len(cand_array)
    norms_sq = np.sum(cand_array ** 2, axis=1)
    
    # Greedy selection with dynamic max_norm tracking
    # Key fix: track the actual max norm of selected vectors, not all candidates
    selected = []
    selected_norms_sq = []
    current_max_norm_sq = 0
    
    # Sort by norm ascending to pack low-norm vectors first
    order = np.argsort(norms_sq)
    
    for idx in order:
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        
        v = cand_array[idx]
        v_norm_sq = norms_sq[idx]
        
        # The constraint for adding v: 
        # 1. dist(v, sel) >= sqrt(max(current_max_norm_sq, v_norm_sq)) for all sel
        # 2. After adding, new max_norm_sq = max(current_max_norm_sq, v_norm_sq)
        
        new_max_norm_sq = max(current_max_norm_sq, v_norm_sq)
        
        # Check compatibility with all selected vectors
        compatible = True
        for sel_idx in selected:
            sel_v = cand_array[sel_idx]
            diff = v - sel_v
            dist_sq = np.sum(diff ** 2)
            if dist_sq < new_max_norm_sq:
                compatible = False
                break
        
        if compatible:
            selected.append(idx)
            selected_norms_sq.append(v_norm_sq)
            current_max_norm_sq = new_max_norm_sq
    
    if not selected:
        return _seed_solution()
    
    result = cand_array[selected]
    
    # Verify the solution
    if len(result) > 1:
        result_norms_sq = np.sum(result ** 2, axis=1)
        max_norm_sq = np.max(result_norms_sq)
        # Check all pairwise distances
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                diff = result[i] - result[j]
                dist_sq = np.sum(diff ** 2)
                if dist_sq < max_norm_sq:
                    # Invalid - fall back to seed
                    return _seed_solution()
    
    return result
# EVOLVE-BLOCK-END
