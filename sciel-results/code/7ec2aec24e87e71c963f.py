import numpy as np
import time
import itertools
import random

D = 11

def construct():
    start_time = time.time()
    deadline = 140.0

    # Generate candidate pool
    candidates = []
    
    # Norm squared 1: ±e_i (22 vectors)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(tuple(v))
    
    # Norm squared 2: ±e_i ± e_j (220 vectors)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(tuple(v))
    
    # Norm squared 3: three ±1s (C(11,3) * 8 = 165 * 8 = 1320 vectors)
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(tuple(v))
    
    # Norm squared 4: (±2,0,...) or (±1,±1,±1,±1) etc.
    # (±2, 0, ...) : 22 vectors
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s * 2
            candidates.append(tuple(v))
    
    # (±1,±1,±1,±1,0,...) : C(11,4)*16 = 330*16 = 5280 vectors - too many, skip
    # (±1,±1,0,...) with one zero? No, that's norm 2.
    # (±2,±1,0,...) : norm 5, skip for now
    
    # Norm squared 5: (±1,±2,0,...) permutations
    # C(11,2) * 2 * 4 = 55 * 8 = 440 vectors
    for i, j in itertools.permutations(range(D), 2):
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = [0] * D
                v[i] = s1
                v[j] = s2 * 2
                candidates.append(tuple(v))
    
    # Remove duplicates
    candidates = list(set(candidates))
    n_cands = len(candidates)
    
    cand_array = np.array(candidates, dtype=np.int64)
    norm_sq = np.sum(cand_array * cand_array, axis=1)
    
    # Greedy with multiple random restarts
    best_selected = []
    best_score = 0
    rng = random.Random(42)
    
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Create ordering: sort by norm squared, randomize within same norm
        order = list(range(n_cands))
        # Group by norm
        norm_groups = {}
        for idx in order:
            n = int(norm_sq[idx])
            if n not in norm_groups:
                norm_groups[n] = []
            norm_groups[n].append(idx)
        
        # Randomize within each group
        for n in norm_groups:
            rng.shuffle(norm_groups[n])
        
        # Build order: sort by norm
        ordered = []
        for n in sorted(norm_groups.keys()):
            ordered.extend(norm_groups[n])
        
        # Greedy
        selected = []
        selected_set = set()
        sel_norms = []
        
        for idx in ordered:
            if time.time() > deadline:
                break
            c = cand_array[idx]
            c_set = tuple(c)
            if c_set in selected_set:
                continue
            n2_c = int(norm_sq[idx])
            
            if not selected:
                selected.append(idx)
                selected_set.add(c_set)
                sel_norms.append(n2_c)
                continue
            
            # Check compatibility
            sel_arr = cand_array[selected]
            diffs = sel_arr - c
            dist_sq = np.sum(diffs * diffs, axis=1)
            max_n2 = max(sel_norms)
            if np.min(dist_sq) >= max_n2:
                selected.append(idx)
                selected_set.add(c_set)
                sel_norms.append(n2_c)
        
        if len(selected) > best_score:
            best_score = len(selected)
            best_selected = selected
    
    # Local search on best solution
    if best_selected:
        mask = np.zeros(n_cands, dtype=bool)
        for idx in best_selected:
            mask[idx] = True
        
        # Try to add more
        unselected = np.where(~mask)[0]
        for _ in range(100):
            if time.time() > deadline:
                break
            if len(unselected) == 0:
                break
            added_any = False
            for _ in range(20):
                idx = unselected[rng.randint(0, len(unselected) - 1)]
                c = cand_array[idx]
                sel_idx = np.where(mask)[0]
                if len(sel_idx) == 0:
                    mask[idx] = True
                    added_any = True
                    break
                sel_arr = cand_array[sel_idx]
                diffs = sel_arr - c
                dist_sq = np.sum(diffs * diffs, axis=1)
                max_n2 = int(np.max(norm_sq[sel_idx]))
                if np.min(dist_sq) >= max_n2:
                    mask[idx] = True
                    added_any = True
                    break
            if not added_any:
                unselected = np.where(~mask)[0]
                if len(unselected) == 0:
                    break
        
        best_selected = list(np.where(mask)[0])
        best_score = len(best_selected)
    
    result = cand_array[best_selected]
    return result.astype(np.int64)
