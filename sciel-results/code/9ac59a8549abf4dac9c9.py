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
        # Direct sum with tetracode-like structure
        # This is a generic fallback, not optimized for specific n
        # For n=8, we use the specific solver above
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # This fallback is weak for large n, but the specific case is handled
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute neighbors
    # neighbors[w] is a list of 17 indices: w itself + 16 neighbors
    neighbors = np.zeros((total, 17), dtype=np.int32)
    
    # Precompute weights for base-3 conversion
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    for w in range(total):
        neighbors[w, 0] = w
        idx = 1
        temp = w
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors[w, idx] = new_w
                idx += 1
    
    best_size = total
    best_set = None
    
    # We will run multiple iterations of Greedy + Local Search
    # Local Search: Removal + Swap (1-for-1)
    
    iteration = 0
    while time.time() < deadline - 2.0:
        iteration += 1
        
        # 1. Randomized Greedy Construction
        # To get diversity, randomize the initial order or tie-breaking.
        
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        uncovered_count = total
        
        # Optimization: Use a list of candidates? Or just scan?
        # Scanning 6561 items is fast.
        
        while uncovered_count > 0:
            # Count how many uncovered words each candidate covers
            # counts[c] = sum(uncovered_mask[neighbors[c]])
            counts = uncovered_mask[neighbors].sum(axis=1)
            
            max_count = np.max(counts)
            
            if max_count == 0:
                break
            
            # Find all candidates with max count
            candidates = np.where(counts == max_count)[0]
            
            # Pick one randomly
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update cover counts and uncovered mask
            ball = neighbors[best_idx]
            was_uncovered = uncovered_mask[ball]
            num_newly_covered = np.sum(was_uncovered)
            uncovered_count -= num_newly_covered
            
            cover_count[ball] += 1
            uncovered_mask[ball] = False
            
            # Check time
            if time.time() > deadline - 5.0:
                break
        
        if uncovered_count > 0:
            continue
        
        # 2. Local Search: Removal and Swap
        
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            
            # Pass 1: Removal
            # A codeword c can be removed if all words in B(c) are covered by at least one other codeword.
            # i.e., for all w in B(c), cover_count[w] > 1.
            
            # min_counts[c] = min(cover_count[neighbors[c]])
            min_counts = np.min(cover_count[neighbors], axis=1)
            
            # Find removable codewords in selected
            removed = False
            for i in range(len(selected)):
                cw = selected[i]
                if min_counts[cw] > 1:
                    # Remove it
                    selected.pop(i)
                    cover_count[neighbors[cw]] -= 1
                    changed = True
                    removed = True
                    break  # Restart loop to recompute min_counts
            
            if not removed:
                # No removals possible. Try Swaps.
                if time.time() < deadline - 5.0:
                    swap_done = False
                    # Try to find a swap for any codeword
                    # To avoid bias, we can iterate through selected in random order
                    order = np.random.permutation(len(selected))
                    
                    for idx in order:
                        if swap_done:
                            break
                        cw = selected[idx]
                        
                        # Identify words that would become uncovered if we remove cw
                        # These are words w in B(cw) such that cover_count[w] == 1
                        ball = neighbors[cw]
                        U = ball[cover_count[ball] == 1]
                        
                        if len(U) == 0:
                            # This should have been removed in the previous pass
                            # But due to race conditions or logic flow, handle it
                            selected.pop(idx)
                            cover_count[ball] -= 1
                            changed = True
                            swap_done = True
                            break
                            
                        if len(U) > 3:
                            # Cannot be covered by a single codeword
                            # (Max size of intersection of balls of radius 1 is limited)
                            # Specifically, if U has 4 words, they must be within dist 2 of each other?
                            # Actually, a codeword c' covers U if dist(c', w) <= 1 for all w in U.
                            # This implies dist(w1, w2) <= 2 for all w1, w2 in U.
                            # In ternary length 8, this is possible for small sets.
                            # But if |U| is large, it's unlikely.
                            # Let's be conservative: only try if |U| <= 3.
                            continue
                        
                        # We need a c' not in selected such that U is subset of B(c')
                        # To find c', we can check candidates.
                        # Efficient check:
                        # For each w in U, c' must be in B(w).
                        # So c' must be in the intersection of B(w) for w in U.
                        
                        # Let's find the intersection of the balls of words in U.
                        # B(w) has 17 elements.
                        # Intersection of 3 sets of 17 elements is small.
                        
                        # Get candidates from the first word in U
                        candidates = neighbors[U[0]]
                        
                        # Filter candidates
                        valid_candidates = []
                        for c_prime in candidates:
                            # Check if c_prime covers all words in U
                            # c_prime covers w if w is in neighbors[c_prime]
                            # Or dist(c_prime, w) <= 1
                            # We can check if U is subset of neighbors[c_prime]
                            # This is O(|U|) check using numpy sum
                            if np.all(neighbors[c_prime] == U): # This is wrong, neighbors[c_prime] is a set of 17, U is a subset
                                pass
                            
                            # Correct check:
                            # count how many words in U are in neighbors[c_prime]
                            # We can use a lookup or just loop since |U| <= 3
                            is_valid = True
                            for w in U:
                                # Check if w is in neighbors[c_prime]
                                # neighbors[c_prime] is an array of 17 ints
                                # We can check if w is in the list
                                # Since 17 is small, a linear scan or np.isin is fine
                                if w not in neighbors[c_prime]:
                                    is_valid = False
                                    break
                            
                            if is_valid and c_prime not in selected:
                                valid_candidates.append(c_prime)
                        
                        if valid_candidates:
                            # Pick a random valid candidate
                            c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                            
                            # Perform swap
                            selected[idx] = c_prime
                            # Update cover counts
                            cover_count[neighbors[cw]] -= 1
                            cover_count[neighbors[c_prime]] += 1
                            
                            changed = True
                            swap_done = True
                            break
                            
                    if not swap_done:
                        # No swaps found. Try Perturbation.
                        if time.time() < deadline - 10.0 and len(selected) > 486:
                            # Perturbation: Remove a random codeword, then repair
                            orig_selected = selected.copy()
                            orig_cover_count = cover_count.copy()
                            
                            idx_to_remove = np.random.randint(len(selected))
                            cw_removed = selected.pop(idx_to_remove)
                            cover_count[neighbors[cw_removed]] -= 1
                            
                            # Find uncovered words
                            U = np.where(cover_count == 0)[0]
                            
                            if len(U) > 0:
                                # Greedy repair
                                repair_count = 0
                                while len(U) > 0 and repair_count < 5:
                                    # Find best codeword to cover U
                                    u_mask = np.zeros(total, dtype=bool)
                                    u_mask[U] = True
                                    counts = u_mask[neighbors].sum(axis=1)
                                    
                                    # Exclude already selected
                                    counts[selected] = 0
                                    
                                    best_cw = np.argmax(counts)
                                    
                                    if counts[best_cw] == 0:
                                        break
                                        
                                    selected.append(best_cw)
                                    cover_count[neighbors[best_cw]] += 1
                                    
                                    # Update U
                                    U = np.where(cover_count == 0)[0]
                                    repair_count += 1
                                    
                                # If we added more than we removed, revert
                                if len(selected) > len(orig_selected):
                                    selected = orig_selected
                                    cover_count = orig_cover_count
                                    changed = False
                                    break
                                else:
                                    # If size is same or smaller, continue local search
                                    changed = True
                            else:
                                # U is empty, removal was safe (should have been caught earlier)
                                pass

        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
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
