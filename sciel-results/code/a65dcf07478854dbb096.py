import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 585

    # Build a pool of candidate vectors with norm^2 in {2, 4, 6, 8, 10}
    # A vector is valid if it is non-zero and has integer entries.
    # We'll use vectors with small weights to maximize packing.
    
    candidates = []
    cand_support = []
    cand_norm2 = []
    
    # Weight 2 vectors: C(11,2) * 4 = 55 * 4 = 220
    for pos in combinations(range(d), 2):
        for signs in product([-1, 1], repeat=2):
            v = np.zeros(d, dtype=np.int64)
            for i, p in enumerate(pos):
                v[p] = signs[i]
            candidates.append(v)
            cand_support.append(tuple(pos))
            cand_norm2.append(2)
    
    # Weight 3 vectors: C(11,3) * 8 = 165 * 8 = 1320
    for pos in combinations(range(d), 3):
        for signs in product([-1, 1], repeat=3):
            v = np.zeros(d, dtype=np.int64)
            for i, p in enumerate(pos):
                v[p] = signs[i]
            candidates.append(v)
            cand_support.append(tuple(pos))
            cand_norm2.append(3)
    
    # Weight 4 vectors: C(11,4) * 16 = 330 * 16 = 5280
    for pos in combinations(range(d), 4):
        for signs in product([-1, 1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for i, p in enumerate(pos):
                v[p] = signs[i]
            candidates.append(v)
            cand_support.append(tuple(pos))
            cand_norm2.append(4)
    
    # Weight 5 vectors: C(11,5) * 32 = 462 * 32 = 14784 - too many, skip
    # Actually let's include weight 5 but with a subset
    # For now, stick with weights 2, 3, 4
    
    n_cand = len(candidates)
    
    # Precompute squared norms
    norms2 = np.array(cand_norm2, dtype=np.int64)
    
    # For the kissing condition: min squared distance >= max squared norm
    # squared distance between i and j = ||vi||^2 + ||vj||^2 - 2*vi.vj
    # We need: ||vi||^2 + ||vj||^2 - 2*vi.vj >= max_norm2 for all pairs
    # where max_norm2 = max(||vk||^2 for k in selected)
    
    # Since all our vectors have norm^2 in {2, 3, 4}, max_norm2 = 4 (if any weight-4 is selected)
    # or 3 (if only weight 2,3) or 2 (if only weight 2)
    
    # Strategy: try to maximize k. The condition is:
    # For all pairs i,j in selected: ||vi-vj||^2 >= max_norm2
    # ||vi-vj||^2 = ||vi||^2 + ||vj||^2 - 2*vi·vj
    
    # Precompute dot products between all pairs? That's n_cand^2 which is too large.
    # Instead, we'll use a greedy approach with incremental checking.
    
    # Group by support for faster conflict checking
    support_to_cands = {}
    for i, s in enumerate(cand_support):
        if s not in support_to_cands:
            support_to_cands[s] = []
        support_to_cands[s].append(i)
    
    support_list = list(support_to_cands.keys())
    support_to_idx = {s: i for i, s in enumerate(support_list)}
    n_supports = len(support_list)
    
    def squared_dist(i, j):
        """Compute squared distance between candidates i and j."""
        vi = candidates[i]
        vj = candidates[j]
        return norms2[i] + norms2[j] - 2 * np.dot(vi, vj)
    
    def can_add(idx, selected_list, max_norm2):
        """Check if candidate idx can be added given current selection."""
        for sel in selected_list:
            if squared_dist(idx, sel) < max_norm2:
                return False
        return True
    
    def greedy_random(rng, max_weight=4):
        """Greedy with random ordering, restricted to vectors with weight <= max_weight."""
        # Filter candidates by weight
        if max_weight == 2:
            valid_indices = [i for i in range(n_cand) if cand_norm2[i] == 2]
        elif max_weight == 3:
            valid_indices = [i for i in range(n_cand) if cand_norm2[i] <= 3]
        else:
            valid_indices = list(range(n_cand))
        
        order = rng.permutation(len(valid_indices))
        selected = []
        selected_set = set()
        max_norm2 = 2  # Start with minimum possible
        
        for oi in order:
            idx = valid_indices[oi]
            # Check if adding this would violate the condition
            # New max_norm2 would be max(max_norm2, norms2[idx])
            new_max = max(max_norm2, int(norms2[idx]))
            
            ok = True
            for sel in selected:
                if squared_dist(idx, sel) < new_max:
                    ok = False
                    break
            
            if ok:
                selected.append(idx)
                selected_set.add(idx)
                if int(norms2[idx]) > max_norm2:
                    max_norm2 = int(norms2[idx])
                    # When max_norm2 increases, we need to re-verify all existing pairs
                    # Actually, the condition is min_dist >= max_norm2
                    # If max_norm2 increases, we need to check that all existing pairs still satisfy
                    # Let's re-verify
                    all_ok = True
                    for i in range(len(selected)):
                        for j in range(i+1, len(selected)):
                            if squared_dist(selected[i], selected[j]) < max_norm2:
                                all_ok = False
                                break
                        if not all_ok:
                            break
                    if not all_ok:
                        # This is a problem - we need to backtrack
                        # For simplicity, just don't add it
                        selected.pop()
                        selected_set.discard(idx)
                        ok = False
        
        return selected
    
    def local_search(selected, rng, time_limit):
        """Local search: try to add points, and if stuck, swap."""
        selected_set = set(selected)
        max_norm2 = max(int(norms2[i]) for i in selected) if selected else 2
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < time_limit:
            improved = False
            
            # Try to add any point
            added_any = False
            for idx in range(n_cand):
                if idx not in selected_set:
                    new_max = max(max_norm2, int(norms2[idx]))
                    ok = True
                    for sel in selected:
                        if squared_dist(idx, sel) < new_max:
                            ok = False
                            break
                    if ok:
                        # Also verify existing pairs if max_norm2 increases
                        if new_max > max_norm2:
                            for i in range(len(selected)):
                                for j in range(i+1, len(selected)):
                                    if squared_dist(selected[i], selected[j]) < new_max:
                                        ok = False
                                        break
                                if not ok:
                                    break
                        
                        if ok:
                            selected.append(idx)
                            selected_set.add(idx)
                            max_norm2 = new_max
                            improved = True
                            added_any = True
                            break
            
            if added_any:
                no_improve_count = 0
                continue
            
            # Stuck: try swap
            if len(selected) > 0:
                # Pick a random point to remove
                remove_idx = rng.integers(0, len(selected))
                removed = selected[remove_idx]
                selected.pop(remove_idx)
                selected_set.discard(removed)
                
                # Recompute max_norm2
                if selected:
                    max_norm2 = max(int(norms2[i]) for i in selected)
                else:
                    max_norm2 = 2
                
                # Try to add multiple points
                additions = 0
                for idx in range(n_cand):
                    if idx not in selected_set:
                        new_max = max(max_norm2, int(norms2[idx]))
                        ok = True
                        for sel in selected:
                            if squared_dist(idx, sel) < new_max:
                                ok = False
                                break
                        if ok:
                            if new_max > max_norm2:
                                for i in range(len(selected)):
                                    for j in range(i+1, len(selected)):
                                        if squared_dist(selected[i], selected[j]) < new_max:
                                            ok = False
                                            break
                                        if not ok:
                                            break
                            if ok:
                                selected.append(idx)
                                selected_set.add(idx)
                                max_norm2 = new_max
                                additions += 1
                
                if additions > 0:
                    no_improve_count = 0
                else:
                    # Put it back
                    selected.append(removed)
                    selected_set.add(removed)
                    max_norm2 = max(int(norms2[i]) for i in selected)
                    no_improve_count += 1
                
                if no_improve_count > 30:
                    break
        
        return selected
    
    best_points = []
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Greedy with different weights and random seeds
    for max_w in [2, 3, 4]:
        for trial in range(10):
            if time.time() > deadline - 60:
                break
            selected = greedy_random(rng, max_weight=max_w)
            if len(selected) > best_size:
                best_size = len(selected)
                best_points = selected[:]
    
    # Phase 2: Local search from best
    if time.time() < deadline - 30 and len(best_points) > 0:
        ls_result = local_search(best_points[:], rng, deadline - 60)
        if len(ls_result) > best_size:
            best_size = len(ls_result)
            best_points = ls_result[:]
    
    # Phase 3: More restarts with perturbation
    if time.time() < deadline - 30:
        for _ in range(5):
            if time.time() >= deadline - 30:
                break
            start = best_points[:]
            for _ in range(3):
                if len(start) > 0:
                    ri = rng.integers(0, len(start))
                    start.pop(ri)
            
            # Greedy fill
            start_set = set(start)
            max_norm2 = max(int(norms2[i]) for i in start) if start else 2
            order = rng.permutation(n_cand)
            for idx in order:
                if idx not in start_set:
                    new_max = max(max_norm2, int(norms2[idx]))
                    ok = True
                    for sel in start:
                        if squared_dist(idx, sel) < new_max:
                            ok = False
                            break
                    if ok:
                        if new_max > max_norm2:
                            for i in range(len(start)):
                                for j in range(i+1, len(start)):
                                    if squared_dist(start[i], start[j]) < new_max:
                                        ok = False
                                        break
                                    if not ok:
                                        break
                        if ok:
                            start.append(idx)
                            start_set.add(idx)
                            max_norm2 = new_max
            
            ls_result = local_search(start, rng, deadline - 30)
            if len(ls_result) > best_size:
                best_size = len(ls_result)
                best_points = ls_result[:]
    
    if len(best_points) < 2:
        points = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
    else:
        points = [candidates[i] for i in best_points]
    
    pts_array = np.array(points, dtype=np.int64)
    return np.around(pts_array).astype(np.int64)
