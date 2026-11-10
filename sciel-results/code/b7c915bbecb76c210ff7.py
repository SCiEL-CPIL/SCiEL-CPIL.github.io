import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 560
    
    base_points = []
    for i in range(d):
        p1 = np.zeros(d, dtype=np.int64)
        p1[i] = 2
        base_points.append(p1)
        p2 = np.zeros(d, dtype=np.int64)
        p2[i] = -2
        base_points.append(p2)
    
    combos4 = list(itertools.combinations(range(d), 4))
    combos5 = list(itertools.combinations(range(d), 5))
    all_combos = combos4 + combos5
    n_combos = len(all_combos)
    weights = np.array([16]*len(combos4) + [32]*len(combos5), dtype=np.int64)
    
    combo_masks = np.array([sum(1 << idx for idx in c) for c in all_combos], dtype=np.int32)
    
    # Precompute conflict matrix
    conflict = np.zeros((n_combos, n_combos), dtype=np.bool_)
    for i in range(n_combos):
        for j in range(i+1, n_combos):
            if bin(combo_masks[i] & combo_masks[j]).count('1') > 2:
                conflict[i,j] = True
                conflict[j,i] = True
    
    best_selected = np.zeros(n_combos, dtype=np.bool_)
    best_weight = 0
    
    n_trials = 500
    for trial in range(n_trials):
        if time.time() > deadline - 150:
            break
        
        if trial == 0:
            order = np.arange(n_combos)
        else:
            rng = np.random.default_rng(trial * 104729 + 7)
            order = rng.permutation(n_combos)
        
        selected = np.zeros(n_combos, dtype=np.bool_)
        for ci in order:
            if not np.any(conflict[ci] & selected):
                selected[ci] = True
        
        sel_weight = weights[selected].sum()
        if sel_weight > best_weight:
            best_weight = sel_weight
            best_selected = selected.copy()
    
    # Local search
    if time.time() < deadline - 30:
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            # Try adding
            candidates = np.where(~best_selected)[0]
            for cj in candidates:
                if not np.any(conflict[cj] & best_selected):
                    best_selected[cj] = True
                    improved = True
                    break
            # Try swap
            if not improved:
                selected_idx = np.where(best_selected)[0]
                for si in selected_idx:
                    if time.time() > deadline - 15:
                        break
                    best_selected[si] = False
                    for cj in candidates:
                        if best_selected[cj]:
                            continue
                        if not np.any(conflict[cj] & best_selected):
                            new_w = weights[best_selected].sum() + weights[cj]
                            if new_w > best_weight:
                                best_selected[cj] = True
                                best_weight = new_w
                                improved = True
                            else:
                                best_selected[si] = True
                            break
                    if not improved:
                        best_selected[si] = True
                    if improved:
                        break
    
    points = list(base_points)
    for ci in np.where(best_selected)[0]:
        c = all_combos[ci]
        n = len(c)
        for mask in range(1 << n):
            p = np.zeros(d, dtype=np.int64)
            for k, idx in enumerate(c):
                p[idx] = -1 if (mask >> k) & 1 else 1
            points.append(p)
    
    return np.array(points, dtype=np.int64)
