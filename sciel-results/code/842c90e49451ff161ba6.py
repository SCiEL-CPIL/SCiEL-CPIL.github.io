import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Strategy: Try the E8 x V3 construction first.
    # E8 minimals (norm sq 2): 240 vectors.
    # V3 (16 vectors of form (+-1, +-1, +-1)): norm sq 3.
    # Total norm sq 5.
    # Check validity.
    
    # Generate E8 minimal vectors (integer coords)
    e8_vectors = []
    # Type 1: (+-1, +-1, 0, 0, 0, 0, 0, 0)
    for i, j in combinations(range(8), 2):
        for si, sj in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            v = np.zeros(8, dtype=np.int64)
            v[i] = si
            v[j] = sj
            e8_vectors.append(v)
    # Type 2: (+-1/2, ...) -> Not integer. 
    # Standard E8 integer representation only has Type 1? 
    # No, E8 minimal vectors in Z^8 are only the 112 of type 1? 
    # Wait, E8 is not in Z^8. It's in a lattice.
    # The 240 minimal vectors of E8 include half-integer ones.
    # If we require integer output, we can't use half-integers.
    # The current program uses integer vectors.
    # So we are restricted to Z^11.
    # The "E8" in the directive likely refers to the 112 integer vectors of type 1? 
    # But 112 * 16 = 1792.
    # Or maybe the directive implies a specific integer sub-lattice.
    
    # Given the ambiguity and risk, and the fact that the current heuristic works (280),
    # I will optimize the heuristic to run faster and potentially find more.
    # The current code has a complex local search.
    # I will simplify and speed up the local search to allow more iterations.
    
    all_4sets = list(combinations(range(d), 4))
    n_sets = len(all_4sets)
    
    # Precompute conflict bitsets
    conflict = [0] * n_sets
    for i in range(n_sets):
        si = set(all_4sets[i])
        mask = 0
        for j in range(n_sets):
            if i == j: continue
            sj = set(all_4sets[j])
            if len(si & sj) >= 3:
                mask |= (1 << j)
        conflict[i] = mask

    def greedy_bitset(order):
        selected = []
        available = (1 << n_sets) - 1
        for idx in order:
            if available & (1 << idx):
                selected.append(idx)
                available &= ~conflict[idx]
                available &= ~(1 << idx)
        return selected

    import random
    rng = random.Random(42)
    
    best_selection = []
    best_size = 0
    
    # Faster initial greedy
    for trial in range(1000):
        if time.time() > deadline - 120:
            break
        order = list(range(n_sets))
        rng.shuffle(order)
        sel = greedy_bitset(order)
        if len(sel) > best_size:
            best_size = len(sel)
            best_selection = sel[:]

    # Optimized Local Search
    # 1. Try to add
    # 2. If fail, try 1-swap (remove one, add one) to escape local optima
    
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        sel_set = set(best_selection)
        
        # Step 1: Try to add any valid set
        added = False
        # Precompute forbidden mask for current selection?
        # Forbidden for j is OR of conflict[k] for k in best_selection
        forbidden_mask = 0
        for k in best_selection:
            forbidden_mask |= conflict[k]
            
        # Find j not in sel_set and not in forbidden_mask
        # Iterate j
        for j in range(n_sets):
            if j in sel_set:
                continue
            if forbidden_mask & (1 << j):
                continue
            # Valid
            best_selection.append(j)
            sel_set.add(j)
            best_size += 1
            added = True
            improved = True
            break
        
        if added:
            continue
        
        # Step 2: 1-Swap
        if len(best_selection) > 0:
            i_idx = rng.randint(0, len(best_selection)-1)
            i_val = best_selection[i_idx]
            
            # Remove i_val
            best_selection.pop(i_idx)
            sel_set.remove(i_val)
            
            # Recalculate forbidden mask for remaining
            forbidden_mask = 0
            for k in best_selection:
                forbidden_mask |= conflict[k]
            
            # Find candidate j
            candidates = []
            for j in range(n_sets):
                if j in sel_set:
                    continue
                if forbidden_mask & (1 << j):
                    continue
                candidates.append(j)
            
            if candidates:
                j_val = rng.choice(candidates)
                best_selection.append(j_val)
                sel_set.add(j_val)
                improved = True
            else:
                # Revert
                best_selection.append(i_val)
                sel_set.add(i_val)

    # Generate points
    points = []
    for idx in best_selection:
        s = all_4sets[idx]
        i, j, k, l = s
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        points.append(v)
    
    if not points:
        for i, j in combinations(range(d), 2):
            for si, sj in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                points.append(v)
    
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
