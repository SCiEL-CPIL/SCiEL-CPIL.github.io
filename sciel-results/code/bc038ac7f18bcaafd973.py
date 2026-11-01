import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

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
    
    # Precompute neighbors (forward: word -> its 17 ball members)
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

    # Precompute reverse neighbors: for each word, which codewords cover it
    rev_neighbors = [[] for _ in range(total)]
    for c in range(total):
        for w in neighbors[c]:
            rev_neighbors[w].append(c)

    # Precompute neighbor sets for fast membership testing in local search
    neighbor_sets = [set(neighbors[w]) for w in range(total)]

    best_size = total
    best_set = None
    
    # Multiple restarts to escape local minima
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 3.0:
            break
            
        # 1. Randomized Greedy Construction
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        uncovered_count = total
        
        while uncovered_count > 0:
            # Count how many uncovered words each candidate covers
            counts = uncovered_mask[neighbors].sum(axis=1)
            
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            # Add some randomness to tie-breaking to diversify
            # Instead of always picking max, pick from top-k
            top_k = min(5, total)
            candidates = np.where(counts >= max_count - 1)[0]
            if len(candidates) > top_k:
                candidates = candidates[np.random.choice(len(candidates), size=top_k, replace=False)]
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
        
        # 2. Local Search: Removal and Swap
        changed = True
        no_improve_count = 0
        
        while changed and time.time() < deadline - 2.0:
            changed = False
            
            # Pass 1: Removal - find all codewords that are removable
            min_counts = np.min(cover_count[neighbors], axis=1)
            
            # Find removable codewords among selected
            removable_in_selected = [c for c in selected if min_counts[c] > 1]
            
            if len(removable_in_selected) > 0:
                # Remove one at a time to be safe
                cw = removable_in_selected[0]
                selected.remove(cw)
                cover_count[neighbors[cw]] -= 1
                changed = True
                no_improve_count = 0
                continue  # Recompute min_counts
            
            # No removals possible. Try Swaps.
            if time.time() < deadline - 10.0 and len(selected) > 486:
                # Try multiple swap attempts per iteration
                swap_attempted = False
                for _ in range(10):
                    if time.time() > deadline - 3.0:
                        break
                    
                    # Perturbation: Remove a random codeword, then try to find a replacement
                    orig_selected = selected.copy()
                    orig_cover_count = cover_count.copy()
                    
                    idx_to_remove = np.random.randint(len(selected))
                    cw_removed = selected.pop(idx_to_remove)
                    cover_count[neighbors[cw_removed]] -= 1
                    
                    # Find words that became uncovered
                    U = np.where(cover_count == 0)[0]
                    
                    if len(U) > 0:
                        # Find a replacement codeword that covers all words in U
                        candidate_set = set(rev_neighbors[U[0]])
                        
                        for w in U[1:]:
                            candidate_set &= set(rev_neighbors[w])
                            if len(candidate_set) == 0:
                                break
                        
                        # Filter out already-selected and the removed codeword
                        valid_candidates = [c for c in candidate_set if c != cw_removed and c not in selected]
                        
                        replacement_found = False
                        if len(valid_candidates) > 0:
                            # Pick a random valid candidate
                            cw2 = valid_candidates[np.random.randint(len(valid_candidates))]
                            selected.append(cw2)
                            cover_count[neighbors[cw2]] += 1
                            replacement_found = True
                        else:
                            # No single replacement covers all of U.
                            # Try greedy repair: add best codeword to cover most of U
                            u_mask = np.zeros(total, dtype=bool)
                            u_mask[U] = True
                            u_counts = u_mask[neighbors].sum(axis=1)
                            # Exclude already selected
                            for c in selected:
                                u_counts[c] = 0
                            u_counts[cw_removed] = 0
                            
                            best_cw = np.argmax(u_counts)
                            if u_counts[best_cw] > 0:
                                selected.append(best_cw)
                                cover_count[neighbors[best_cw]] += 1
                                # Continue repairing
                                remaining = np.where(cover_count == 0)[0]
                                while len(remaining) > 0 and time.time() < deadline - 3.0:
                                    rem_mask = np.zeros(total, dtype=bool)
                                    rem_mask[remaining] = True
                                    rem_counts = rem_mask[neighbors].sum(axis=1)
                                    for c in selected:
                                        rem_counts[c] = 0
                                    bc = np.argmax(rem_counts)
                                    if rem_counts[bc] == 0:
                                        break
                                    selected.append(bc)
                                    cover_count[neighbors[bc]] += 1
                                    remaining = np.where(cover_count == 0)[0]
                                replacement_found = True
                            else:
                                # Can't repair, revert
                                pass
                        
                        if not replacement_found:
                            selected = orig_selected
                            cover_count = orig_cover_count
                        else:
                            # Check if we improved (should be same size, but maybe enables further removals)
                            # Try to remove redundant codewords after the swap
                            min_counts = np.min(cover_count[neighbors], axis=1)
                            removable_in_selected = [c for c in selected if min_counts[c] > 1]
                            if len(removable_in_selected) > 0:
                                cw = removable_in_selected[0]
                                selected.remove(cw)
                                cover_count[neighbors[cw]] -= 1
                                changed = True
                                no_improve_count = 0
                                swap_attempted = True
                                break
                    else:
                        # U is empty - the removal didn't break coverage
                        changed = True
                        no_improve_count = 0
                        swap_attempted = True
                        break
                    
                    # Revert if no improvement
                    selected = orig_selected
                    cover_count = orig_cover_count
                
                if not swap_attempted:
                    no_improve_count += 1
                    if no_improve_count > 5:
                        break
        # End Local Search

        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            no_improve_count = 0
            
        if best_size <= 486:
            break

    # Convert best_set to list of digit lists
    if best_set is None:
        # Fallback
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
