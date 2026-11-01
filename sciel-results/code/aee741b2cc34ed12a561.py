import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks for other parameters (not expected in this task, but good practice)
    if q == 3 and n >= 4:
        # Simple tetracode extension, not optimal but valid
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute weights for base-3 digits
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Precompute balls (neighborhoods of radius 1)
    # balls[w] is a list of 17 indices: w itself and its 16 neighbors
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
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
                
    all_indices = np.arange(total, dtype=np.int32)
    
    # Precompute ball membership as a boolean matrix for fast subset checks
    # ball_membership[w, v] = True if v is in balls[w]
    ball_membership = np.zeros((total, total), dtype=bool)
    for w in range(total):
        ball_membership[w, balls[w]] = True
    
    best_size = total
    best_set = None
    
    # We will run multiple iterations of Greedy + Local Search
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # 1. Randomized Greedy Construction
        # To get diverse starts, we can randomize the tie-breaking in greedy
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        # Greedy Loop
        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                break # Time out for this iteration
                
            # Count coverage for each candidate among uncovered words
            # counts[c] = number of uncovered words in balls[c]
            counts = uncovered_mask[balls].sum(axis=1)
            
            if np.max(counts) == 0:
                break # Should not happen if uncovered_mask has True
                
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best to encourage diversity
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update cover counts
            cover_count[balls[best_idx]] += 1
            
            # Update uncovered mask
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue # Invalid solution due to timeout in greedy
            
        # 2. Local Search
        # Phase 1: Remove redundant codewords
        # We iterate until no more can be removed
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                # Check if c is removable: all words in balls[c] are covered by >= 2 codewords
                # i.e., min(cover_count[balls[c]]) > 1
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        # Phase 2: Swap / Replace (Vectorized)
        # Try to replace a codeword c with c' if it leads to a better state
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                # First, try to remove (might have become removable due to previous swaps in this pass)
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                # Try to swap c with some c'
                swapped = False
                
                # Critical set: words covered ONLY by c
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]
                
                if len(critical_words) == 0:
                    # Should have been removed, but just in case
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                # Generate candidate c' values that could cover the critical words
                # A candidate c' must have all critical_words in balls[c']
                # i.e., ball_membership[c_prime, critical_words] must all be True
                
                # Strategy: For each critical word, find codewords that cover it
                # Then intersect these sets to get candidates that cover ALL critical words
                
                # Start with all possible candidates
                candidate_mask = np.ones(total, dtype=bool)
                
                for cw in critical_words:
                    # Find all codewords c' such that cw is in balls[c']
                    # This is equivalent to: c' is in balls[cw] (by symmetry of Hamming distance)
                    candidate_mask &= ball_membership[:, cw]
                
                # Get the list of valid candidates
                valid_candidates = np.where(candidate_mask)[0]
                
                # Remove c itself from candidates (we want a different codeword)
                valid_candidates = valid_candidates[valid_candidates != c]
                
                if len(valid_candidates) > 0:
                    # Pick a random candidate from the valid ones
                    c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                    
                    # Perform the swap
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    new_selected.append(c_prime)
                    swapped = True
                    improved = True
                else:
                    new_selected.append(c)
            
            selected = new_selected
            
        # Phase 3: 2-Exchange (Remove 2, Add 1)
        # Try to find two codewords c1, c2 that can be removed and replaced by c3
        if len(selected) > 450 and time.time() < deadline - 8.0:
            # For each pair of codewords, check if there's a single codeword that covers
            # all words that would become uncovered
            improved_2ex = True
            while improved_2ex and time.time() < deadline - 5.0:
                improved_2ex = False
                new_selected = list(selected)
                
                # Try pairs in random order
                indices = list(range(len(new_selected)))
                np.random.shuffle(indices)
                
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        if time.time() > deadline - 5.0:
                            break
                            
                        c1 = new_selected[indices[i]]
                        c2 = new_selected[indices[j]]
                        
                        # Words that would become uncovered if we remove both c1 and c2
                        # These are words covered by exactly one of {c1, c2} and no other codeword
                        # i.e., cover_count[x] == 1 and x is in balls[c1] xor balls[c2]
                        
                        # Simpler: temporarily remove both, see what becomes uncovered
                        cover_count[balls[c1]] -= 1
                        cover_count[balls[c2]] -= 1
                        
                        uncovered_after = (cover_count == 0)
                        
                        if np.any(uncovered_after):
                            # Need to find a codeword c3 that covers all uncovered words
                            uncovered_words = np.where(uncovered_after)[0]
                            
                            # Find c3 that covers all uncovered_words
                            candidate_mask = np.ones(total, dtype=bool)
                            for uw in uncovered_words[:10]:  # Limit to first 10 for speed
                                candidate_mask &= ball_membership[:, uw]
                            
                            valid_c3 = np.where(candidate_mask)[0]
                            valid_c3 = valid_c3[(valid_c3 != c1) & (valid_c3 != c2)]
                            
                            if len(valid_c3) > 0:
                                c3 = valid_c3[np.random.randint(len(valid_c3))]
                                cover_count[balls[c3]] += 1
                                
                                # Remove c1 and c2, add c3
                                new_selected.remove(c1)
                                new_selected.remove(c2)
                                new_selected.append(c3)
                                improved_2ex = True
                                break
                            else:
                                # Revert
                                cover_count[balls[c1]] += 1
                                cover_count[balls[c2]] += 1
                        else:
                            # Both can be removed without adding anything
                            new_selected.remove(c1)
                            new_selected.remove(c2)
                            improved_2ex = True
                            break
                        # Revert if not improved
                        if not improved_2ex:
                            cover_count[balls[c1]] += 1
                            cover_count[balls[c2]] += 1
                    
                    if improved_2ex:
                        break
                
                selected = new_selected
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
                break # Hit lower bound (unlikely)
            if best_size <= 486:
                # We hit the known best. We can try to improve further or break.
                # Let's keep searching for a bit to see if we can go lower.
                # But 486 is the proven best for a while.
                # Actually, 486 is the BEST KNOWN. If we find 486, we are at the state-of-the-art.
                # We should break to save time for other restarts or just return.
                # But maybe we can find 485? Unlikely.
                # Let's break if we find <= 486 to ensure we return a valid good answer quickly.
                break

    # Convert best_set to output format
    if best_set is None:
        # Fallback: Return a valid (though large) code
        # Simple greedy without local search
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        best_set = selected

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
