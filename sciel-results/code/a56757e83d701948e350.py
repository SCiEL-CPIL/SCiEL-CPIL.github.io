import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580
    
    # Weight-4 vectors
    four_positions = list(combinations(range(d), 4))
    n_pos = len(four_positions)  # 330
    
    # Create all candidate vectors
    candidates = []
    cand_support = []
    for pos in four_positions:
        for signs in product([-1, 1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for i, p in enumerate(pos):
                v[p] = signs[i]
            candidates.append(v)
            cand_support.append(tuple(pos))
    
    n_cand = len(candidates)  # 5280
    
    # Group candidates by support
    support_to_cands = {}
    for i, s in enumerate(cand_support):
        if s not in support_to_cands:
            support_to_cands[s] = []
        support_to_cands[s].append(i)
    
    # Precompute same-support conflicts
    same_support_conflict = {}
    for s, cands in support_to_cands.items():
        conflict_map = {i: set() for i in cands}
        for i in range(len(cands)):
            for j in range(i+1, len(cands)):
                vi = candidates[cands[i]]
                vj = candidates[cands[j]]
                ham = sum(1 for p in s if vi[p] != vj[p])
                if ham <= 1:
                    conflict_map[cands[i]].add(cands[j])
                    conflict_map[cands[j]].add(cands[i])
        same_support_conflict[s] = conflict_map
    
    # Precompute supports with intersection size 3
    support_list = list(support_to_cands.keys())
    support_to_idx = {s: i for i, s in enumerate(support_list)}
    n_supports = len(support_list)
    
    # For each pair of supports with intersection 3, precompute conflicting candidate pairs
    inter3_conflict = {}
    for i in range(n_supports):
        s1_set = set(support_list[i])
        for j in range(i+1, n_supports):
            s2_set = set(support_list[j])
            inter = s1_set & s2_set
            if len(inter) == 3:
                inter_tuple = tuple(sorted(inter))
                cands1 = support_to_cands[support_list[i]]
                cands2 = support_to_cands[support_list[j]]
                conflict_set = set()
                for i1 in cands1:
                    v1 = candidates[i1]
                    for i2 in cands2:
                        v2 = candidates[i2]
                        if all(v1[p] == v2[p] for p in inter_tuple):
                            conflict_set.add((i1, i2))
                            conflict_set.add((i2, i1))
                inter3_conflict[(i, j)] = conflict_set
    
    def can_add(idx, selected_set, selected_list):
        """Check if candidate idx can be added to the current set."""
        s_idx = support_to_idx[cand_support[idx]]
        for sel in selected_list:
            sel_s_idx = support_to_idx[cand_support[sel]]
            if s_idx == sel_s_idx:
                if sel in same_support_conflict[cand_support[idx]][idx]:
                    return False
            else:
                key = (min(s_idx, sel_s_idx), max(s_idx, sel_s_idx))
                if key in inter3_conflict:
                    if (idx, sel) in inter3_conflict[key]:
                        return False
        return True
    
    def greedy_random(rng):
        """Run a greedy pass with random ordering."""
        order = rng.permutation(n_cand)
        selected = []
        selected_set = set()
        for idx in order:
            if can_add(idx, selected_set, selected):
                selected.append(idx)
                selected_set.add(idx)
        return selected
    
    def local_search(selected, rng, time_limit):
        """Iteratively try to add points; if stuck, remove a point to unblock."""
        selected_set = set(selected)
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < time_limit:
            improved = False
            
            # Try to add any point
            added_any = False
            for idx in range(n_cand):
                if idx not in selected_set:
                    if can_add(idx, selected_set, selected):
                        selected.append(idx)
                        selected_set.add(idx)
                        improved = True
                        added_any = True
                        break
            
            if added_any:
                no_improve_count = 0
                continue
            
            # Stuck: try removing one point to allow adding more
            if len(selected) > 0:
                remove_idx = rng.integers(0, len(selected))
                removed = selected[remove_idx]
                selected.pop(remove_idx)
                selected_set.discard(removed)
                
                # Try to add multiple points after removal
                additions = 0
                for idx in range(n_cand):
                    if idx not in selected_set:
                        if can_add(idx, selected_set, selected):
                            selected.append(idx)
                            selected_set.add(idx)
                            additions += 1
                
                # Only keep removal if it led to net gain
                if additions > 0:
                    no_improve_count = 0
                else:
                    # Put it back
                    selected.append(removed)
                    selected_set.add(removed)
                    no_improve_count += 1
                
                if no_improve_count > 50:
                    break
        
        return selected
    
    def swap_search(selected, rng, time_limit):
        """Try to swap one point for another that allows more additions."""
        selected_set = set(selected)
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < time_limit:
            improved = False
            
            # Try to add any point
            added_any = False
            for idx in range(n_cand):
                if idx not in selected_set:
                    if can_add(idx, selected_set, selected):
                        selected.append(idx)
                        selected_set.add(idx)
                        improved = True
                        added_any = True
                        break
            
            if added_any:
                no_improve_count = 0
                continue
            
            # Try a swap: remove one point, add one that was previously blocked
            if len(selected) > 0:
                # Try a few random swaps
                for attempt in range(10):
                    remove_idx = rng.integers(0, len(selected))
                    removed = selected[remove_idx]
                    
                    # Find a candidate that was blocked by 'removed' but can be added after removal
                    # Temporarily remove
                    selected.pop(remove_idx)
                    selected_set.discard(removed)
                    
                    # Try to add a new point
                    added = -1
                    for idx in range(n_cand):
                        if idx not in selected_set:
                            if can_add(idx, selected_set, selected):
                                added = idx
                                break
                    
                    if added >= 0:
                        selected.append(added)
                        selected_set.add(added)
                        no_improve_count = 0
                        improved = True
                        break
                    else:
                        # Put it back
                        selected.append(removed)
                        selected_set.add(removed)
                
                if not improved:
                    no_improve_count += 1
                if no_improve_count > 100:
                    break
        
        return selected
    
    best_points = []
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Multiple greedy restarts
    num_greedy = 0
    while time.time() < deadline - 150:
        num_greedy += 1
        selected = greedy_random(rng)
        if len(selected) > best_size:
            best_size = len(selected)
            best_points = selected[:]
        if best_size >= 125:
            break
    
    # Phase 2: Local search from best
    if time.time() < deadline - 100 and len(best_points) > 0:
        ls_result = local_search(best_points[:], rng, deadline - 100)
        if len(ls_result) > best_size:
            best_size = len(ls_result)
            best_points = ls_result[:]
    
    # Phase 3: Swap search
    if time.time() < deadline - 80 and len(best_points) > 0:
        swap_result = swap_search(best_points[:], rng, deadline - 80)
        if len(swap_result) > best_size:
            best_size = len(swap_result)
            best_points = swap_result[:]
    
    # Phase 4: More local search rounds
    if time.time() < deadline - 60:
        for _ in range(10):
            if time.time() >= deadline - 60:
                break
            start = best_points[:]
            for _ in range(3):
                if len(start) > 0:
                    ri = rng.integers(0, len(start))
                    start.pop(ri)
            
            start_set = set(start)
            order = rng.permutation(n_cand)
            for idx in order:
                if idx not in start_set:
                    if can_add(idx, start_set, start):
                        start.append(idx)
                        start_set.add(idx)
            
            ls_result = local_search(start, rng, deadline - 60)
            if len(ls_result) > best_size:
                best_size = len(ls_result)
                best_points = ls_result[:]
    
    if len(best_points) < 2:
        points = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
    else:
        points = [candidates[i] for i in best_points]
    
    pts_array = np.array(points, dtype=np.int64)
    return np.around(pts_array).astype(np.int64)
