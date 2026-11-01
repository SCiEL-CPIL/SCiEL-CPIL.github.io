import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_iterated(deadline)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
        
    # Generic fallback: return all words (valid format, worst case)
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_iterated(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls for all words
    # balls[w, 0] = w, balls[w, 1:17] = neighbors
    # Using big-endian weights for index calculation
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
                
    best_size = total
    best_selected = None
    rng = np.random.default_rng(42) # Fixed seed for reproducibility in the search process
    
    # Precompute for faster local search: for each word, store its ball as a tuple for hashing
    ball_tuples = [tuple(balls[i]) for i in range(total)]
    
    # We will run multiple iterations of Randomized Greedy + Local Search
    while time.time() < deadline - 10.0:
        
        # 1. Randomized Greedy Construction
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        current_uncovered = np.ones(total, dtype=bool)
        
        # Greedy Loop with Randomization
        # Instead of always picking the max, pick from top candidates to diversify
        while np.any(current_uncovered):
            if time.time() > deadline - 15.0:
                break # Time out for this iteration, skip to next if valid, else break outer
            
            # Count uncovered neighbors for each word
            counts = np.sum(current_uncovered[balls], axis=1)
            max_count = np.max(counts)
            
            if max_count == 0:
                break
            
            # Get candidates with max count
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best candidates
            # This is the key difference from pure deterministic greedy
            best_idx = candidates[rng.integers(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)
        
        # If timed out during greedy, we might have an invalid partial solution.
        # Complete it quickly to ensure validity for the local search phase.
        if np.any(current_uncovered):
            if time.time() > deadline - 10.0:
                break
            # Simple completion: pick any uncovered word
            while np.any(current_uncovered):
                uncovered_indices = np.where(current_uncovered)[0]
                if len(uncovered_indices) == 0:
                    break
                best_idx = uncovered_indices[0]
                selected.append(best_idx)
                cover_count[balls[best_idx]] += 1
                current_uncovered = (cover_count == 0)
                if time.time() > deadline - 10.0:
                    break
        
        # 2. Aggressive Pruning (Greedy Removal)
        # Remove codewords that are no longer necessary
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            new_selected = []
            for c in selected:
                # Check if removing c is safe: all words in its ball are covered by others
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        
        # 3. Enhanced Local Search: Multiple swap passes with better candidate selection
        # Try to replace codewords to allow further removals
        improved = True
        max_swap_rounds = 10
        round_count = 0
        while improved and time.time() < deadline - 8.0 and round_count < max_swap_rounds:
            improved = False
            round_count += 1
            
            # Build a set for O(1) lookup
            selected_set = set(selected)
            
            # Try all possible single swaps
            for i in range(len(selected)):
                if time.time() > deadline - 8.0:
                    break
                c = selected[i]
                
                # Try replacing c with each word in its ball
                for w in balls[c]:
                    if w in selected_set:
                        continue
                    
                    # Temporarily remove c
                    cover_count[balls[c]] -= 1
                    
                    # Check if coverage is maintained
                    if np.any(cover_count == 0):
                        cover_count[balls[c]] += 1  # Restore
                        continue
                    
                    # Add w
                    cover_count[balls[w]] += 1
                    
                    # Try to remove additional redundant codewords
                    removals_found = []
                    for j in range(len(selected)):
                        if j == i:
                            continue
                        d = selected[j]
                        if np.min(cover_count[balls[d]]) > 1:
                            removals_found.append(d)
                    
                    if removals_found:
                        # Remove all found redundant codewords
                        for d in removals_found:
                            cover_count[balls[d]] -= 1
                            selected_set.remove(d)
                        
                        # Update selected list: replace c with w, remove all d's
                        new_selected = []
                        for idx, codeword in enumerate(selected):
                            if idx == i:
                                new_selected.append(w)
                            elif codeword in removals_found:
                                continue
                            else:
                                new_selected.append(codeword)
                        
                        selected = new_selected
                        selected_set.add(w)
                        improved = True
                        break
                    else:
                        # No improvement, restore state
                        cover_count[balls[w]] -= 1
                        cover_count[balls[c]] += 1
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            # After each round, do a quick pruning pass
            if improved:
                changed = True
                while changed and time.time() < deadline - 8.0:
                    changed = False
                    new_selected = []
                    for c in selected:
                        if np.min(cover_count[balls[c]]) > 1:
                            cover_count[balls[c]] -= 1
                            changed = True
                        else:
                            new_selected.append(c)
                    selected = new_selected

        # 4. Final Pruning Pass after swaps
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected

        # Update Best
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected.copy()
            
            if best_size <= 486:
                # We hit the known optimum (or close to it). 
                # 486 is the best known. If we get 486, we can stop early to save time?
                # Actually, we might get 485? Unlikely. 
                # Let's stop if we reach 486 to be safe and fast.
                break

    # Fallback if best_selected is None (should not happen if loop runs at least once)
    if best_selected is None:
        # Return a valid but large set
        best_selected = list(range(total))
        
    # Convert to digit lists using precomputed weights for efficiency
    result = []
    for w in best_selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
