import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    import itertools
    if q == 3 and n >= 4:
        # Direct sum with tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute neighbors
    neighbors = np.zeros((total, 17), dtype=np.int32)
    
    for w in range(total):
        neighbors[w, 0] = w
        idx = 1
        temp = w
        for pos in range(n):
            weight = 3 ** (n - 1 - pos)
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors[w, idx] = new_w
                idx += 1
    
    # Precompute: for each codeword, a set (as boolean array) of its neighbors for fast membership
    # We'll use the neighbors array directly with boolean indexing
    
    best_size = total
    best_set = None
    
    # Precompute a set of selected for O(1) membership (will be updated)
    
    while time.time() < deadline - 2.0:
        # 1. Randomized Greedy Construction
        
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        selected_set = set()
        uncovered_mask = np.ones(total, dtype=bool)
        uncovered_count = total
        
        while uncovered_count > 0:
            counts = uncovered_mask[neighbors].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            
            ball = neighbors[best_idx]
            was_uncovered = uncovered_mask[ball]
            num_newly_covered = np.sum(was_uncovered)
            uncovered_count -= num_newly_covered
            
            cover_count[ball] += 1
            uncovered_mask[ball] = False
            
            if time.time() > deadline - 5.0:
                break
        
        if uncovered_count > 0:
            continue
        
        # 2. Local Search: Vectorized Removal + Fast Perturbation
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            
            # Vectorized removal pass: find all removable codewords at once
            min_counts = np.min(cover_count[neighbors], axis=1)
            # For selected codewords, check if removable
            sel_arr = np.array(selected, dtype=np.int32)
            if len(sel_arr) > 0:
                removable_mask = min_counts[sel_arr] > 1
                if np.any(removable_mask):
                    # Remove all removable codewords at once
                    removable_indices = np.where(removable_mask)[0]
                    # Process in reverse to avoid index issues
                    for i in removable_indices[::-1]:
                        cw = sel_arr[i]
                        cover_count[neighbors[cw]] -= 1
                    # Update selected list
                    remove_set = set(sel_arr[removable_indices].tolist())
                    selected = [cw for cw in selected if cw not in remove_set]
                    selected_set -= remove_set
                    changed = True
                    continue  # Recompute min_counts
            
            # No removals possible. Try perturbation.
            if time.time() < deadline - 8.0 and len(selected) > 486:
                # Perturbation: Remove 1-3 random codewords, then repair
                orig_selected = selected[:]
                orig_cover_count = cover_count.copy()
                orig_selected_set = selected_set.copy()
                
                num_remove = min(random.randint(1, 3), len(selected))
                removed_cws = []
                for _ in range(num_remove):
                    idx = np.random.randint(len(selected))
                    cw = selected.pop(idx)
                    selected_set.discard(cw)
                    cover_count[neighbors[cw]] -= 1
                    removed_cws.append(cw)
                
                U = np.where(cover_count == 0)[0]
                
                if len(U) > 0:
                    # Fast repair: greedy add best codewords
                    repair_count = 0
                    max_repairs = 4
                    while len(U) > 0 and repair_count < max_repairs:
                        u_mask = np.zeros(total, dtype=bool)
                        u_mask[U] = True
                        counts = u_mask[neighbors].sum(axis=1)
                        # Exclude already selected
                        counts[np.array(list(selected_set), dtype=np.int32)] = 0
                        best_cw = np.argmax(counts)
                        if counts[best_cw] == 0:
                            break
                        selected.append(best_cw)
                        selected_set.add(best_cw)
                        cover_count[neighbors[best_cw]] += 1
                        U = np.where(cover_count == 0)[0]
                        repair_count += 1
                    
                    # If we couldn't fully repair or added too many, revert
                    if len(U) > 0 or len(selected) > len(orig_selected):
                        selected = orig_selected
                        cover_count = orig_cover_count
                        selected_set = orig_selected_set
                else:
                    # U was empty, removal was safe — keep it
                    pass
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
        if best_size <= 486:
            break

    # Convert best_set to list of digit lists
    if best_set is None:
        return [[0] * n]
    
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
