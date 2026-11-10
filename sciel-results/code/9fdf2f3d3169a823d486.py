import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit
    
    # Add Type A points: 2*e_i and -2*e_i
    base_points = []
    for i in range(d):
        p1 = np.zeros(d, dtype=np.int64)
        p1[i] = 2
        base_points.append(p1)
        
        p2 = np.zeros(d, dtype=np.int64)
        p2[i] = -2
        base_points.append(p2)
    
    # Generate all 4-combinations of indices
    combos = list(itertools.combinations(range(d), 4))
    n_combos = len(combos)
    
    # Precompute intersection sizes using bitmask for speed
    combo_masks = []
    for c in combos:
        mask = 0
        for idx in c:
            mask |= (1 << idx)
        combo_masks.append(mask)
    
    # Precompute intersection sizes matrix
    inter_sizes = np.zeros((n_combos, n_combos), dtype=np.int8)
    for i in range(n_combos):
        for j in range(i + 1, n_combos):
            inter = combo_masks[i] & combo_masks[j]
            sz = bin(inter).count('1')
            inter_sizes[i, j] = sz
            inter_sizes[j, i] = sz
            
    # Precompute neighbors (conflicts) for faster greedy
    neighbors = [[] for _ in range(n_combos)]
    for i in range(n_combos):
        for j in range(i + 1, n_combos):
            if inter_sizes[i, j] > 2:
                neighbors[i].append(j)
                neighbors[j].append(i)
    
    best_selected = None
    best_count = 0
    
    # Try multiple random orderings with greedy selection
    n_trials = 5000
    for trial in range(n_trials):
        if time.time() > deadline - 60: # Leave 60s for local search
            break
            
        if trial == 0:
            # Sort by ascending degree
            degrees = [len(neighbors[i]) for i in range(n_combos)]
            order = sorted(range(n_combos), key=lambda x: degrees[x])
        elif trial == 1:
            # Sort by descending degree
            degrees = [len(neighbors[i]) for i in range(n_combos)]
            order = sorted(range(n_combos), key=lambda x: -degrees[x])
        else:
            rng = np.random.default_rng(trial * 7919 + 13)
            order = rng.permutation(n_combos)
        
        selected = []
        forbidden = set()
        
        for ci in order:
            if ci not in forbidden:
                selected.append(ci)
                # Add all neighbors to forbidden
                for nb in neighbors[ci]:
                    forbidden.add(nb)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected[:]
    
    # Local improvement: try to replace one selected combo with an unselected one
    if best_selected is not None and best_count > 0:
        if time.time() < deadline - 10:
            best_selected = best_selected[:]
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                best_set = set(best_selected)
                for si in range(len(best_selected)):
                    if time.time() > deadline - 5:
                        break
                    removed = best_selected[si]
                    remaining = best_set - {removed}
                    for cj in range(n_combos):
                        if cj in remaining:
                            continue
                        valid = True
                        for sj in remaining:
                            if inter_sizes[cj, sj] > 2:
                                valid = False
                                break
                        if valid:
                            best_selected[si] = cj
                            best_set = set(best_selected)
                            improved = True
                            break
                    if improved:
                        break
    
    # Build points array
    points = list(base_points)
    
    if best_selected is not None:
        for ci in best_selected:
            c = combos[ci]
            indices = list(c)
            for mask in range(16):
                p = np.zeros(d, dtype=np.int64)
                for k, idx in enumerate(indices):
                    if (mask >> k) & 1:
                        p[idx] = -1
                    else:
                        p[idx] = 1
                points.append(p)
    
    points_array = np.array(points, dtype=np.int64)
    
    return points_array
