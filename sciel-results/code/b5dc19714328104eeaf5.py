import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
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
    num_neighbors = 1 + n * (q - 1)  # 17
    
    # Precompute neighbors
    neighbors = np.zeros((total, num_neighbors), dtype=np.int32)
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
    
    # We will run multiple iterations of: Greedy -> Local Search
    while time.time() < deadline - 5.0:
        # 1. Randomized Greedy Construction
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        uncovered_count = total
        
        while uncovered_count > 0:
            if time.time() > deadline - 10.0:
                break
            
            # Efficient count: sum of uncovered_mask over the ball of each candidate
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
        
        if uncovered_count > 0:
            continue
        
        # 2. Local Search (Removal + Swaps)
        changed = True
        local_iter = 0
        while changed and time.time() < deadline - 5.0:
            changed = False
            local_iter += 1
            
            # Pass 1: Removal
            # Check if any codeword is redundant (all its neighbors have count > 1)
            # We iterate in random order to break symmetry
            indices_to_check = list(range(len(selected)))
            np.random.shuffle(indices_to_check)
            
            removed_any = False
            for i in indices_to_check:
                if time.time() > deadline - 10.0:
                    break
                cw = selected[i]
                # If min count in its ball is > 1, it's redundant
                if np.min(cover_count[neighbors[cw]]) > 1:
                    selected.pop(i)
                    cover_count[neighbors[cw]] -= 1
                    changed = True
                    removed_any = True
                    break # Restart removal pass
            
            if not changed and len(selected) > 486:
                # Pass 2: Swaps (Perturbation)
                # Try to replace a codeword to potentially allow further removals
                # or to find a better local minimum.
                # Strategy: Pick a random codeword, remove it, find a replacement that covers
                # all the "critical" words (words that would become uncovered).
                
                if time.time() < deadline - 15.0:
                    orig_selected = selected.copy()
                    orig_cover_count = cover_count.copy()
                    
                    idx_to_remove = np.random.randint(len(selected))
                    cw_removed = selected.pop(idx_to_remove)
                    cover_count[neighbors[cw_removed]] -= 1
                    
                    U = np.where(cover_count == 0)[0]
                    
                    if len(U) > 0:
                        # Find replacement using intersection of neighbor sets
                        # A valid replacement must cover all words in U.
                        # The set of codewords covering a word v is neighbors[v] (by symmetry).
                        # So we need the intersection of neighbors[v] for all v in U.
                        
                        # Start with the set of neighbors of the first uncovered word
                        # Convert to a set for fast intersection
                        candidate_set = set(neighbors[U[0]].tolist())
                        
                        # Intersect with neighbors of other uncovered words
                        for u in U[1:]:
                            if not candidate_set:
                                break
                            candidate_set &= set(neighbors[u].tolist())
                        
                        # Exclude currently selected codewords (and the one we removed? 
                        # Actually, we can re-add the removed one, but that's a no-op swap.
                        # Usually we want a NEW codeword to potentially change the structure.
                        # But allowing the same one back is safe if it's the only option.
                        # However, if we just put it back, we haven't changed the set.
                        # Let's exclude selected ones to force a change, unless no other option.
                        
                        selected_mask = np.zeros(total, dtype=bool)
                        selected_mask[selected] = True
                        
                        valid_candidates = []
                        for c in candidate_set:
                            if not selected_mask[c]:
                                valid_candidates.append(c)
                        
                        if len(valid_candidates) > 0:
                            # Pick one randomly
                            cw2 = valid_candidates[np.random.randint(len(valid_candidates))]
                            selected.append(cw2)
                            cover_count[neighbors[cw2]] += 1
                            changed = True # Allow further removals in next iteration
                        else:
                            # No replacement found, revert
                            selected = orig_selected
                            cover_count = orig_cover_count
                    else:
                        # U is empty, removal was valid (should have been caught in removal pass)
                        # But if it happens, we improved size.
                        changed = True
            
            # Pass 3: 2-exchange local search (try swapping two codewords)
            if not changed and len(selected) > 486 and time.time() < deadline - 20.0:
                # Try a 2-exchange: remove two codewords, find two replacements
                # This can escape local minima that single swaps can't
                if random.random() < 0.3:  # Do this with some probability to limit overhead
                    orig_selected = selected.copy()
                    orig_cover_count = cover_count.copy()
                    
                    # Pick two random indices
                    i1, i2 = np.random.choice(len(selected), size=2, replace=False)
                    cw1 = selected.pop(i1)
                    cw2 = selected.pop(max(0, i2 - (1 if i2 > i1 else 0)))
                    
                    cover_count[neighbors[cw1]] -= 1
                    cover_count[neighbors[cw2]] -= 1
                    
                    U = np.where(cover_count == 0)[0]
                    
                    if len(U) <= 4:  # Only try if few words become uncovered
                        # Find two replacements that together cover all of U
                        # Try all pairs from the candidate set
                        if len(U) == 0:
                            # Both removals were valid
                            changed = True
                        else:
                            # Get candidates for each uncovered word
                            # We need a pair (a, b) such that for each u in U, 
                            # a covers u OR b covers u
                            
                            # Start with candidates from the first uncovered word
                            cand1 = set(neighbors[U[0]].tolist())
                            for u in U[1:]:
                                cand1 &= set(neighbors[u].tolist())
                            
                            # For each candidate a in cand1, check if there's a b that covers the rest
                            # Actually, we need: for each u in U, a in neighbors[u] OR b in neighbors[u]
                            # This is equivalent to: the set of words NOT covered by a must be covered by b
                            
                            found_pair = False
                            for a in list(cand1)[:20]:  # Limit to 20 candidates for speed
                                # Words not covered by a
                                not_covered_by_a = [u for u in U if a not in set(neighbors[u].tolist())]
                                if len(not_covered_by_a) == 0:
                                    # a covers all, we just need any b (but we need to remove 2 and add 2)
                                    # Actually if a covers all, we can pick any b that doesn't create issues
                                    # But we need to add exactly 2 codewords
                                    # Let's just pick a random valid b
                                    b_candidates = []
                                    for b in range(total):
                                        if b not in selected and b != a:
                                            # Check that adding a and b doesn't break anything
                                            # Actually we already removed cw1 and cw2, so we just need
                                            # that a and b together cover U
                                            # Since a covers all of U, any b works as long as it's valid
                                            b_candidates.append(b)
                                            if len(b_candidates) > 10:
                                                break
                                    if b_candidates:
                                        b = b_candidates[random.randint(0, len(b_candidates) - 1)]
                                        selected.append(a)
                                        selected.append(b)
                                        cover_count[neighbors[a]] += 1
                                        cover_count[neighbors[b]] += 1
                                        found_pair = True
                                        changed = True
                                        break
                                elif len(not_covered_by_a) == 1:
                                    # b must cover not_covered_by_a[0]
                                    u_rem = not_covered_by_a[0]
                                    b_candidates = [b for b in set(neighbors[u_rem].tolist()) 
                                                    if b not in selected and b != a]
                                    if b_candidates:
                                        b = b_candidates[random.randint(0, len(b_candidates) - 1)]
                                        # Verify that a and b together cover all of U
                                        ok = True
                                        for u in U:
                                            if a not in set(neighbors[u].tolist()) and b not in set(neighbors[u].tolist()):
                                                ok = False
                                                break
                                        if ok:
                                            selected.append(a)
                                            selected.append(b)
                                            cover_count[neighbors[a]] += 1
                                            cover_count[neighbors[b]] += 1
                                            found_pair = True
                                            changed = True
                                            break
                                else:
                                    # More than 1 word not covered by a, skip for now
                                    continue
                            
                            if not found_pair:
                                # Revert
                                selected = orig_selected
                                cover_count = orig_cover_count
                    else:
                        # Too many uncovered words, revert
                        selected = orig_selected
                        cover_count = orig_cover_count
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            # Final polish: try to remove more codewords from the best solution
            if best_size <= 500 and time.time() < deadline - 10.0:
                polish_cover = np.zeros(total, dtype=np.int32)
                for cw in best_set:
                    polish_cover[neighbors[cw]] += 1
                
                polish_selected = best_set.copy()
                polish_changed = True
                while polish_changed and time.time() < deadline - 5.0:
                    polish_changed = False
                    for i in range(len(polish_selected)):
                        if time.time() > deadline - 5.0:
                            break
                        cw = polish_selected[i]
                        if np.min(polish_cover[neighbors[cw]]) > 1:
                            polish_selected.pop(i)
                            polish_cover[neighbors[cw]] -= 1
                            polish_changed = True
                            break
                
                if len(polish_selected) < best_size:
                    best_size = len(polish_selected)
                    best_set = polish_selected
        
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
