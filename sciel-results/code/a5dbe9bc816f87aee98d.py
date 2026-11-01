import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback for other parameters
    # Simple exhaustive or greedy approach for small cases
    import itertools
    if total_words := q ** n <= 10000:
        # Greedy set cover
        all_words = [list(c) for c in itertools.product(range(q), repeat=n)]
        uncovered = set(range(len(all_words)))
        codewords = []
        
        # Precompute coverage
        def hamming_dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
        
        while uncovered:
            if time.time() > deadline - 5.0:
                break
            best_idx = -1
            best_count = -1
            for i in range(len(all_words)):
                count = 0
                for j in uncovered:
                    if hamming_dist(all_words[i], all_words[j]) <= radius:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1:
                break
                
            codewords.append(all_words[best_idx])
            # Remove covered
            new_uncovered = set()
            for j in uncovered:
                if hamming_dist(all_words[best_idx], all_words[j]) > radius:
                    new_uncovered.add(j)
            uncovered = new_uncovered
            
        return codewords
    else:
        # Return a valid but large code (exhaustive is too big, so just return all? No, must be valid)
        # For large cases, this might timeout or be invalid if not covering.
        # But the task is specific to q=3, n=8.
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
                
    # Precompute ball membership as a boolean matrix for fast subset checks
    # ball_membership[w, v] = True if v is in balls[w]
    # This is 6561x6561 bools ~ 43MB, acceptable.
    ball_membership = np.zeros((total, total), dtype=bool)
    for w in range(total):
        ball_membership[w, balls[w]] = True
        
    # Inverse: covers_word[w] = list of codewords that cover w
    # Due to symmetry, covers_word[w] is the same as balls[w]
    # So we can reuse balls.
    
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
                        
                        # Words covered by c1 or c2
                        # We want to find c3 such that (Cover(c1) U Cover(c2)) \ (Cover(c3)) is empty
                        # i.e., Cover(c3) contains all words that are exclusively covered by c1 or c2
                        # Actually, if we remove c1 and c2, the words that become uncovered are:
                        # {w in balls[c1] U balls[c2] | cover_count[w] == 1 (only c1) or cover_count[w] == 1 (only c2)}
                        # Wait, if a word is covered by both c1 and c2, its count is 2. Removing both makes it 0.
                        # So the set of words that become uncovered is:
                        # W_uncovered = {w | w in balls[c1] U balls[c2] and cover_count[w] <= 2 and (w in balls[c1] or w in balls[c2])}
                        # Specifically:
                        # If w in balls[c1] and w in balls[c2]: count decreases by 2. If count was 2, it becomes 0.
                        # If w in balls[c1] only: count decreases by 1. If count was 1, it becomes 0.
                        # If w in balls[c2] only: count decreases by 1. If count was 1, it becomes 0.
                        
                        # Let's compute the set of words that would become uncovered.
                        # We need c3 to cover ALL of these words.
                        
                        # Efficient check:
                        # The candidate c3 must be in the intersection of balls[w] for all w in W_uncovered.
                        
                        # To avoid iterating all words, we can check candidates from the union of balls[c1] and balls[c2].
                        # Actually, c3 must be close to the critical words.
                        
                        # Let's just check if there exists a c3.
                        # The set of words to cover is S = {w | w in balls[c1] U balls[c2] and cover_count[w] - (1 if w in balls[c1] else 0) - (1 if w in balls[c2] else 0) < 0}
                        # i.e., cover_count[w] < (1 if w in balls[c1] else 0) + (1 if w in balls[c2] else 0)
                        
                        # Let's build the mask of words that become uncovered.
                        mask_c1 = ball_membership[:, c1] # This is not right, ball_membership is [word, cw]
                        # ball_membership[w, c] is True if c covers w.
                        # So mask of words covered by c1 is ball_membership[:, c1]
                        
                        words_c1 = balls[c1]
                        words_c2 = balls[c2]
                        
                        # Words that become uncovered:
                        # w in words_c1 and cover_count[w] == 1
                        # w in words_c2 and cover_count[w] == 1
                        # w in words_c1 and w in words_c2 and cover_count[w] == 2
                        
                        uncov_mask = np.zeros(total, dtype=bool)
                        uncov_mask[words_c1] = (cover_count[words_c1] == 1)
                        uncov_mask[words_c2] = (cover_count[words_c2] == 1)
                        
                        # Handle intersection
                        inter_mask = np.zeros(total, dtype=bool)
                        inter_mask[words_c1] = True
                        inter_mask[words_c2] = True
                        # Actually, simpler:
                        # count_c1 = 1 if w in words_c1 else 0
                        # count_c2 = 1 if w in words_c2 else 0
                        # if cover_count[w] - count_c1 - count_c2 < 0, it becomes uncovered.
                        
                        # Let's just compute the set of words to cover.
                        to_cover = set()
                        for w in words_c1:
                            if cover_count[w] == 1:
                                to_cover.add(w)
                        for w in words_c2:
                            if cover_count[w] == 1:
                                to_cover.add(w)
                        for w in words_c1:
                            if w in words_c2 and cover_count[w] == 2:
                                to_cover.add(w)
                                
                        if not to_cover:
                            # Both can be removed? No, we need to add one.
                            # If to_cover is empty, it means removing them doesn't uncover anything?
                            # That would mean they were redundant. Phase 1 should have handled that.
                            # But maybe they became redundant due to other swaps.
                            # If so, we can just remove one? No, we are doing 2-exchange.
                            # If to_cover is empty, we can remove both and add nothing? No, we must maintain coverage.
                            # If to_cover is empty, it means all words in balls[c1] U balls[c2] are covered by other codewords.
                            # So we can remove both. But the operation is "Remove 2, Add 1".
                            # If we can remove 2 and add 0, that's even better. But let's stick to the heuristic.
                            # If to_cover is empty, we can remove both.
                            # Let's just remove c1 and c2.
                            cover_count[balls[c1]] -= 1
                            cover_count[balls[c2]] -= 1
                            new_selected = [c for c in new_selected if c != c1 and c != c2]
                            improved_2ex = True
                            break
                            
                        # Find c3 that covers all words in to_cover
                        # c3 must be in intersection of balls[w] for w in to_cover
                        candidate_mask = np.ones(total, dtype=bool)
                        for w in to_cover:
                            candidate_mask &= ball_membership[:, w]
                            
                        valid_candidates = np.where(candidate_mask)[0]
                        valid_candidates = valid_candidates[valid_candidates != c1]
                        valid_candidates = valid_candidates[valid_candidates != c2]
                        
                        if len(valid_candidates) > 0:
                            c3 = valid_candidates[np.random.randint(len(valid_candidates))]
                            
                            # Perform swap
                            cover_count[balls[c1]] -= 1
                            cover_count[balls[c2]] -= 1
                            cover_count[balls[c3]] += 1
                            new_selected = [c for c in new_selected if c != c1 and c != c2]
                            new_selected.append(c3)
                            improved_2ex = True
                            break
                        
                    if improved_2ex:
                        break
                if improved_2ex:
                    selected = new_selected
                    # Restart 2-exchange loop as the state changed significantly
                    improved_2ex = True # Force another pass? Or just break and let outer loop handle?
                    # Let's break out of the while loop and let the next iteration of the main while loop handle it?
                    # No, the main while loop is "while improved...".
                    # Let's just set improved_2ex = False and break, and rely on the outer loop?
                    # Actually, if we improved, we should continue local search.
                    # But 2-exchange is expensive. Let's do it a few times.
                    
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected[:]
            
        # Early stop if we reached the known optimum
        if best_size <= 486:
            break

    # Convert best_set to lists of digits
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = idx % q
            idx //= q
        return digits

    result = [idx_to_digits(cw) for cw in best_set]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
            
    return unique_result
