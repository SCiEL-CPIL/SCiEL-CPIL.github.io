"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import random


def construct():
    # Generate candidate pools for different norms
    
    # Pool 1: Norm^2 = 2 (two +/- 1s). Limit dot <= 1
    cand_2 = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    cand_2.append(v)
    cand_2 = np.array(cand_2)
    
    # Pool 2: Norm^2 = 3 (three +/- 1s). Limit dot <= 1
    cand_3 = []
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for signs in product((1, -1), repeat=3):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j], v[k] = signs
                    cand_3.append(v)
    cand_3 = np.array(cand_3)
    
    # Pool 3: Norm^2 = 4 (one +/- 2 OR four +/- 1s). Limit dot <= 2
    cand_4 = []
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            cand_4.append(v)
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for l in range(k+1, D):
                    for signs in product((1, -1), repeat=4):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j], v[k], v[l] = signs
                        cand_4.append(v)
    cand_4 = np.array(cand_4)
    
    def greedy_select(pool, max_dot, shuffle_seed=None):
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
        indices = np.arange(len(pool))
        random.shuffle(indices)
        selected = []
        for idx in indices:
            v = pool[idx]
            ok = True
            for w in selected:
                if np.dot(v, w) > max_dot:
                    ok = False
                    break
            if ok:
                selected.append(v)
        return selected

    def local_search_improve(selected, pool, max_dot, max_iters=1000):
        # Convert selected to list of lists for easy manipulation
        sel = [v.copy() for v in selected]
        # Create a set of hashes for quick lookup? 
        # Vectors are small, so we can just use list comparison or a set of tuples
        sel_set = set(tuple(v) for v in sel)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try to add a random vector from pool
            # Pick a random candidate
            idx = random.randint(0, len(pool) - 1)
            v = pool[idx]
            if tuple(v) in sel_set:
                continue
            # Check if v fits with all in sel
            fits = True
            for w in sel:
                if np.dot(v, w) > max_dot:
                    fits = False
                    break
            if fits:
                sel.append(v)
                sel_set.add(tuple(v))
                improved = True
            else:
                # Try to swap: remove a random existing vector, see if we can add 2 new ones?
                # Or just remove one and try to add one different one that allows more future adds?
                # Simple swap: remove random w from sel, try to add v.
                # If v fits with the rest, accept swap.
                if len(sel) > 1:
                    rem_idx = random.randint(0, len(sel) - 1)
                    w = sel[rem_idx]
                    # Temporarily remove w
                    sel.pop(rem_idx)
                    sel_set.remove(tuple(w))
                    # Check if v fits with new sel
                    fits_v = True
                    for u in sel:
                        if np.dot(v, u) > max_dot:
                            fits_v = False
                            break
                    if fits_v:
                        sel.append(v)
                        sel_set.add(tuple(v))
                        improved = True
                    else:
                        # Revert
                        sel.append(w)
                        sel_set.add(tuple(w))
                        # Try to find ANY vector that fits to replace w?
                        # This is expensive. Just revert.
                        
        return sel

    best_vectors = []
    
    # Run randomized greedy + local search
    for pool, max_dot in [(cand_2, 1), (cand_3, 1), (cand_4, 2)]:
        for seed in range(20):
            sel = greedy_select(pool, max_dot, shuffle_seed=seed)
            # Improve with local search
            sel = local_search_improve(sel, pool, max_dot, max_iters=500)
            if len(sel) > len(best_vectors):
                best_vectors = sel
                # print(f"New best: {len(sel)} from pool {max_dot} seed {seed}")
                
    return np.array(best_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
