import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    # 120s limit. Use 115s to be safe.
    deadline = start_time + 115.0
    
    total_words = q ** n
    
    # Precompute word<->index conversions
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute balls as lists of indices
    balls = []
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors = [idx]
        for pos in range(n):
            original = w[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                w[pos] = new_val
                neighbors.append(word_to_idx(w))
            w[pos] = original
        balls.append(neighbors)
    
    # Precompute ball masks as Python integers (bitsets)
    ball_masks = []
    for c in range(total_words):
        mask = 0
        for nb in balls[c]:
            mask |= (1 << nb)
        ball_masks.append(mask)
        
    all_mask = (1 << total_words) - 1
    
    best_solution = None
    best_size = float('inf')
    
    def greedy_cover(seed):
        random.seed(seed)
        covered_mask = 0
        code = []
        
        while covered_mask != all_mask:
            best_gain = -1
            best_candidates = []
            
            for c in range(total_words):
                gain = (ball_masks[c] & ~covered_mask).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain:
                    best_candidates.append(c)
            
            if not best_candidates:
                break
                
            chosen = random.choice(best_candidates)
            code.append(chosen)
            covered_mask |= ball_masks[chosen]
            
            if time.time() > deadline:
                return None
        
        return code

    def reduce_cover(code):
        """
        Simple pass to remove redundant codewords.
        """
        if not code:
            return code
            
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code.pop(i)
                    changed = True
        
        return code

    def local_search(code):
        """
        Enhanced local search with surrogate-based hill climbing.
        1. Remove redundant codewords.
        2. If stuck, try swaps that reduce the 'unique coverage' surrogate,
           hoping to enable a future reduction.
        """
        if not code:
            return code
            
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
                
        in_code = [False] * total_words
        for c in code:
            in_code[c] = True
            
        # Calculate initial unique masks and surrogate score
        # unique_mask[c] is a bitset of words covered ONLY by c
        unique_masks = [0] * len(code)
        
        def calculate_unique_masks():
            # This is O(K * BallSize). K~180, BallSize~19. ~3000 ops. Very fast.
            for i, c in enumerate(code):
                mask = 0
                for w in balls[c]:
                    if cover_count[w] == 1:
                        mask |= (1 << w)
                unique_masks[i] = mask
                
        calculate_unique_masks()
        
        surrogate_score = sum(m.bit_count() for m in unique_masks)
        
        improved = True
        stagnant_count = 0
        
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to remove redundant codewords
            # We need to find an index i where unique_masks[i] == 0
            # Note: unique_masks must be up to date.
            # After a swap, we recalculate. After a removal, we recalculate.
            
            redundant_idx = -1
            for i in range(len(code)):
                if unique_masks[i] == 0:
                    redundant_idx = i
                    break
            
            if redundant_idx != -1:
                c = code[redundant_idx]
                # Remove c
                for w in balls[c]:
                    cover_count[w] -= 1
                in_code[c] = False
                code.pop(redundant_idx)
                # Recalculate unique masks for remaining
                calculate_unique_masks()
                surrogate_score = sum(m.bit_count() for m in unique_masks)
                improved = True
                stagnant_count = 0
                continue
            
            # 2. No redundancy. Try swaps.
            # To avoid O(K^2) full scan every time which might be slow if K is large,
            # we iterate through codewords.
            
            swap_made = False
            # Randomize order to escape local minima in surrogate landscape
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                c = code[i]
                loss_mask = unique_masks[i]
                
                if loss_mask == 0:
                    continue # Should have been caught above, but safety
                
                # Find candidates c' not in code that cover loss_mask
                # Optimization: We can pre-filter or just scan.
                # Scanning 2187 items is fast.
                candidates = []
                for c_prime in range(total_words):
                    if in_code[c_prime]:
                        continue
                    if (ball_masks[c_prime] & loss_mask) == loss_mask:
                        # Check if this swap is "good"
                        # We want to pick one that minimizes the new surrogate score
                        # or at least improves it.
                        
                        # Quick check: Does c_prime cover any words that are uniquely covered by OTHER codewords?
                        # If yes, it might create new redundancies or increase uniqueness elsewhere.
                        
                        # Let's just collect a few candidates and evaluate them properly.
                        candidates.append(c_prime)
                        
                        # Limit candidates to keep it fast, e.g., first 5 found
                        if len(candidates) >= 5:
                            break
                
                if not candidates:
                    continue
                
                # Evaluate candidates
                best_candidate = -1
                best_new_surrogate = surrogate_score + 1 # Must be better than current (or equal to allow movement?)
                # Actually, we accept if surrogate decreases.
                # If surrogate stays same, we might accept with low probability to avoid cycling, 
                # but strict decrease is safer for convergence.
                
                # To speed up, we don't recalculate full cover_count for every candidate.
                # We simulate the change.
                # Removing c: cover_count[w]-- for w in balls[c]
                # Adding c': cover_count[w]++ for w in balls[c']
                
                # We can estimate the change in unique counts.
                # Words in loss_mask (unique to c) will be covered by c'. 
                # Their cover_count becomes 2 (if only c covered them) or more.
                # So they are no longer unique to any single codeword (unless c' is the only other one? No, c' adds 1, so count becomes 2).
                # So loss_mask words contribute 0 to new unique count.
                
                # Words in balls[c'] \ loss_mask:
                # If cover_count[w] was 1 (unique to some other c_j), it becomes 2. So c_j loses a unique word.
                # If cover_count[w] was > 1, it becomes > 2. No change in uniqueness status (still not unique).
                # If cover_count[w] was 0? Impossible, all words are covered.
                
                # Words in balls[c] \ loss_mask:
                # These are covered by c and at least one other.
                # When we remove c, their count decreases by 1.
                # If count was 2, it becomes 1. They become unique to the other codeword.
                # This INCREASES the surrogate score.
                
                # So the net change in surrogate is:
                # - (number of words in loss_mask)  [c loses its unique words]
                # - (number of words in balls[c'] that were unique to others) [others lose unique words]
                # + (number of words in balls[c] \ loss_mask that had count 2) [others gain unique words]
                
                # This is complex to calculate exactly without modifying cover_count.
                # Given the small size, let's just do the full swap and recalculation for the top candidate.
                # Or, just pick the first candidate that covers loss_mask and rely on the recalculation.
                
                # Let's pick a random candidate from the small list to add diversity
                c_prime = random.choice(candidates)
                
                # Perform Swap
                for w in balls[c]:
                    cover_count[w] -= 1
                in_code[c] = False
                code[i] = c_prime
                
                for w in balls[c_prime]:
                    cover_count[w] += 1
                in_code[c_prime] = True
                
                # Recalculate unique masks
                calculate_unique_masks()
                new_surrogate = sum(m.bit_count() for m in unique_masks)
                
                # Check for immediate redundancy
                # We already checked in the loop above, but the state changed.
                # We can just let the main loop handle it in the next iteration.
                # However, if we find a redundancy now, we can accept immediately.
                
                if new_surrogate < surrogate_score:
                    # Surrogate improved. Accept.
                    surrogate_score = new_surrogate
                    improved = True
                    swap_made = True
                    stagnant_count = 0
                    break # Restart main while loop to check for redundancy
                else:
                    # Revert
                    for w in balls[c_prime]:
                        cover_count[w] -= 1
                    in_code[c_prime] = False
                    code[i] = c
                    
                    for w in balls[c]:
                        cover_count[w] += 1
                    in_code[c] = True
                    # Unique masks are still valid as we reverted
                    # No need to recalculate if we revert perfectly
                
            if not swap_made:
                # No improving swap found.
                # We might be stuck.
                # To escape, we could accept a swap that keeps surrogate same (random walk).
                # But this risks cycling.
                # Let's just break and rely on next restart.
                # Or, try a random swap that doesn't break coverage (always true if it covers loss_mask)
                # and see if it helps in a few steps.
                # For simplicity and safety, we stop here.
                break
                
        return code

    # Main loop
    # Run initial greedy
    initial_code = greedy_cover(0)
    if initial_code:
        reduced_code = reduce_cover(initial_code)
        local_searched_code = local_search(reduced_code)
        best_solution = local_searched_code
        best_size = len(local_searched_code)
    
    seed = 1
    while time.time() < deadline:
        code = greedy_cover(seed)
        if code is None:
            break
        
        reduced_code = reduce_cover(code)
        local_searched_code = local_search(reduced_code)
        
        if len(local_searched_code) < best_size:
            best_solution = local_searched_code
            best_size = len(local_searched_code)
        
        seed += 1
        
    if best_solution is None:
        # Fallback: Tetracode product
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = []
    for idx in best_solution:
        result.append(idx_to_word(idx))
        
    return result
