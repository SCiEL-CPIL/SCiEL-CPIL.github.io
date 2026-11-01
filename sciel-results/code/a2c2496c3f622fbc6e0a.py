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

    best_size = total
    best_set = None
    
    while time.time() < deadline - 2.0:
        # 1. Randomized Greedy Construction
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
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
        
        # 2. Local Search
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            
            # Pass 1: Removal
            min_counts = np.min(cover_count[neighbors], axis=1)
            
            removed_any = False
            indices_to_check = list(range(len(selected)))
            np.random.shuffle(indices_to_check)
            
            for i in indices_to_check:
                cw = selected[i]
                if min_counts[cw] > 1:
                    selected.pop(i)
                    cover_count[neighbors[cw]] -= 1
                    changed = True
                    removed_any = True
                    break
            
            if not changed:
                # Try Swaps (Perturbation)
                if time.time() < deadline - 10.0 and len(selected) > 486:
                    orig_selected = selected.copy()
                    orig_cover_count = cover_count.copy()
                    
                    idx_to_remove = np.random.randint(len(selected))
                    cw_removed = selected.pop(idx_to_remove)
                    cover_count[neighbors[cw_removed]] -= 1
                    
                    U = np.where(cover_count == 0)[0]
                    
                    if len(U) > 0:
                        # Find replacement using vectorized check
                        # We need cw2 such that U is subset of neighbors[cw2]
                        # and cw2 is not in selected and cw2 != cw_removed
                        
                        # Build a mask for valid candidates
                        # valid_mask[cw2] = 1 if U is subset of neighbors[cw2]
                        
                        # Optimization: Only check if U[0] is covered by cw2 first?
                        # No, just vectorize the subset check.
                        
                        # Create a boolean array `is_valid` of size total, initialized to True
                        is_valid = np.ones(total, dtype=bool)
                        
                        # Exclude selected and removed
                        # We can do this later or now.
                        # Let's check the subset condition first.
                        
                        for u in U:
                            # mask_u[cw2] is True if u is in neighbors[cw2]
                            # neighbors is (total, 17). 
                            # (neighbors == u) is (total, 17) bool.
                            # .any(axis=1) is (total,) bool.
                            mask_u = (neighbors == u).any(axis=1)
                            is_valid &= mask_u
                        
                        # Now exclude selected codewords
                        # We need to be careful with `selected` list membership in vectorized op.
                        # Create a boolean array for selected
                        selected_mask = np.zeros(total, dtype=bool)
                        selected_mask[selected] = True
                        is_valid[selected_mask] = False
                        is_valid[cw_removed] = False
                        
                        valid_candidates = np.where(is_valid)[0]
                        
                        if len(valid_candidates) > 0:
                            # Pick one randomly
                            cw2 = valid_candidates[np.random.randint(len(valid_candidates))]
                            selected.append(cw2)
                            cover_count[neighbors[cw2]] += 1
                            changed = True # Allow further removals
                        else:
                            # No replacement found, revert
                            selected = orig_selected
                            cover_count = orig_cover_count
                    else:
                        # U is empty, removal was valid (should have been caught in removal pass)
                        # But if it happens, we improved size.
                        changed = True

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
        if best_size <= 486:
            break

    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n).ljust(n, '0')) for w in range(total)]
        
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
