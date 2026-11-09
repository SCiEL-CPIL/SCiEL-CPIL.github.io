import numpy as np
from itertools import combinations
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 590
    
    # Generate all 4-sets
    all_4sets = list(combinations(range(d), 4))
    n_sets = len(all_4sets)
    
    # Precompute bitmask for each 4-set
    masks = []
    for s in all_4sets:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
        
    # Precompute conflict masks
    # conflict_mask[i] is a bitmask of indices j such that set i and set j share >= 3 indices
    conflict_masks = [0] * n_sets
    for i in range(n_sets):
        mi = masks[i]
        cm = 0
        for j in range(n_sets):
            if i == j:
                continue
            inter = bin(mi & masks[j]).count('1')
            if inter >= 3:
                cm |= (1 << j)
        conflict_masks[i] = cm
        
    best_supports = []
    best_size = 0
    
    rng = random.Random(42)
    
    # Helper to get mask of a set of indices
    def get_mask(idx_set):
        m = 0
        for i in idx_set:
            m |= (1 << i)
        return m

    # Greedy with restarts
    num_tries = 100
    for trial in range(num_tries):
        if time.time() > deadline - 20:
            break
            
        order = list(range(n_sets))
        rng.shuffle(order)
        
        selected = set()
        selected_mask = 0
        
        for idx in order:
            if (conflict_masks[idx] & selected_mask) == 0:
                selected.add(idx)
                selected_mask |= (1 << idx)
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_supports = list(selected)
            
            # Local Search
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                # Try to add a node
                for v in range(n_sets):
                    if v not in selected and (conflict_masks[v] & selected_mask) == 0:
                        selected.add(v)
                        selected_mask |= (1 << v)
                        improved = True
                        best_size = len(selected)
                        best_supports = list(selected)
                
                # Try to swap: remove u, add v
                # This is expensive O(N^2). 
                # Given N=330, N^2 ~ 100k. 
                # If we do this a few times, it's fine.
                if not improved:
                    for u in list(selected):
                        # Tentatively remove u
                        new_mask = selected_mask & ~(1 << u)
                        # Find best v to add
                        best_v = -1
                        # We can just try to add any v that fits
                        # But swapping u for v might allow adding MORE later.
                        # Simple swap: replace u with v if v conflicts with fewer others?
                        # Or just try all v not in selected.
                        for v in range(n_sets):
                            if v in selected:
                                continue
                            if (conflict_masks[v] & new_mask) == 0:
                                # v fits if u is removed
                                # Does this increase size? No, size stays same.
                                # But it might open up space for more.
                                # Let's just do a random swap or skip for now.
                                # A better heuristic: 
                                # Count how many new nodes can be added if we swap u->v.
                                # This is costly.
                                pass
            
    # Generate points
    points = []
    for idx in best_supports:
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
                        
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
