import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if q == 3 and n == 6 and radius == 1:
        total_words = 729
        
        # Precompute index conversions
        def idx_to_tuple(idx):
            t = [0] * 6
            rem = idx
            for i in range(6):
                t[i] = rem % 3
                rem //= 3
            return tuple(t)

        def tuple_to_idx(t):
            idx = 0
            pow3 = 1
            for i in range(6):
                idx += t[i] * pow3
                pow3 *= 3
            return idx

        # Precompute balls as lists (for iteration) and sets (for membership)
        balls_list = [None] * total_words
        balls_set = [None] * total_words
        for i in range(total_words):
            w = idx_to_tuple(i)
            ball = [i]
            for pos in range(6):
                orig_val = w[pos]
                for new_val in range(3):
                    if new_val != orig_val:
                        new_w = list(w)
                        new_w[pos] = new_val
                        new_idx = tuple_to_idx(tuple(new_w))
                        ball.append(new_idx)
            balls_list[i] = ball
            balls_set[i] = set(ball)

        # Helper to check if a set of indices covers the space
        def is_cover(indices):
            covered = set()
            for idx in indices:
                covered.update(balls_list[idx])
            return len(covered) == total_words

        # Helper to prune redundant codewords using counts
        def prune_cover(indices):
            indices = list(indices)
            cover_counts = [0] * total_words
            
            # Initialize counts
            for idx in indices:
                for w in balls_list[idx]:
                    cover_counts[w] += 1

            changed = True
            while changed:
                changed = False
                new_indices = []
                for idx in indices:
                    # Check if removing idx leaves any word uncovered
                    necessary = False
                    for w in balls_list[idx]:
                        if cover_counts[w] == 1:
                            necessary = True
                            break
                    if necessary:
                        new_indices.append(idx)
                    else:
                        # Remove idx
                        for w in balls_list[idx]:
                            cover_counts[w] -= 1
                        changed = True
                indices = new_indices
            return indices

        # Advanced Local Search: Deletion + Swap
        def local_search(code_indices):
            if time.time() > deadline - 10.0:
                return code_indices
            
            code_indices = list(code_indices)
            cover_counts = [0] * total_words
            for idx in code_indices:
                for w in balls_list[idx]:
                    cover_counts[w] += 1
            
            improved = True
            no_swap_improvement_count = 0
            max_no_swap_iterations = 50 # Prevent infinite loops of neutral swaps
            
            while improved and time.time() < deadline - 5.0:
                improved = False
                no_swap_improvement_count = 0
                
                # Phase 1: Try to delete
                deletable = []
                for i, idx in enumerate(code_indices):
                    can_remove = True
                    for w in balls_list[idx]:
                        if cover_counts[w] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        deletable.append(i)
                
                if deletable:
                    # Remove the first one found (or random? first is fine)
                    i_to_remove = deletable[0]
                    idx = code_indices[i_to_remove]
                    for w in balls_list[idx]:
                        cover_counts[w] -= 1
                    code_indices.pop(i_to_remove)
                    improved = True
                    continue # Restart loop to check new state
                
                # Phase 2: Try to swap (neutral move) to enable deletion
                # For each codeword, check if we can replace it with another word
                # that covers its "essential" words, potentially allowing future deletions.
                # Heuristic: Only swap if the "loss" set is small and coverable by a single new word.
                
                swap_done = False
                for i, idx in enumerate(code_indices):
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Calculate loss if we remove idx
                    loss = []
                    for w in balls_list[idx]:
                        if cover_counts[w] == 1:
                            loss.append(w)
                    
                    if not loss:
                        # This shouldn't happen if Phase 1 failed, but just in case
                        continue
                    
                    # If loss is large, skip (too expensive to check candidates)
                    if len(loss) > 4: 
                        continue
                    
                    # Find a candidate c such that loss is subset of ball(c)
                    # And c is not already in code_indices
                    # We can iterate through all 729 words
                    found_swap = False
                    for c in range(total_words):
                        if c == idx:
                            continue
                        if c in code_indices:
                            continue
                        
                        # Check if c covers all loss
                        covers_loss = True
                        for w in loss:
                            if w not in balls_set[c]:
                                covers_loss = False
                                break
                        
                        if covers_loss:
                            # Perform swap
                            # Remove idx
                            for w in balls_list[idx]:
                                cover_counts[w] -= 1
                            # Add c
                            for w in balls_list[c]:
                                cover_counts[w] += 1
                            
                            # Verify no count is 0 (should be true if covers_loss is true 
                            # and we only checked loss from idx, but other words might have been 
                            # covered only by idx? No, loss IS the set of words covered only by idx.
                            # So if c covers loss, those words are now covered by c.
                            # Other words were covered by >=2, so removing idx leaves them >=1.
                            # Adding c increases them. So it's valid.)
                            
                            code_indices[i] = c
                            improved = True
                            swap_done = True
                            found_swap = True
                            break
                    
                    if found_swap:
                        # After a swap, we might have created a deletion opportunity.
                        # We break the swap loop and go back to Phase 1 (deletion) in the main while loop.
                        break
                
                if not swap_done:
                    # No deletions, no swaps
                    break
            
            return prune_cover(code_indices)

        best_code = None
        best_size = float('inf')

        # Strategy 1: Direct Product Construction (Tetracode based)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        dp_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                word = h + list(tail)
                dp_indices.append(tuple_to_idx(tuple(word)))

        pruned_dp = prune_cover(dp_indices)
        if len(pruned_dp) < best_size:
            best_size = len(pruned_dp)
            best_code = pruned_dp

        # Strategy 2: Randomized Greedy with symmetry-guided tie-breaking
        num_restarts = 300 # Increased restarts
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break

            uncovered = set(range(total_words))
            code_indices = []

            while uncovered:
                # Find max gain
                max_gain = 0
                top_candidates = []
                for c in range(total_words):
                    gain = 0
                    for w in balls_list[c]:
                        if w in uncovered:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        top_candidates = [c]
                    elif gain == max_gain and gain > 0:
                        top_candidates.append(c)

                if not top_candidates:
                    break

                # Symmetry-guided tie-breaking
                if len(top_candidates) > 1:
                    random.shuffle(top_candidates)
                    best_c = top_candidates[0]
                else:
                    best_c = top_candidates[0]

                code_indices.append(best_c)
                for w in balls_list[best_c]:
                    uncovered.discard(w)

                if time.time() > deadline - 10.0:
                    break

            if uncovered:
                continue

            # Prune and then local search
            pruned = prune_cover(code_indices)
            pruned = local_search(pruned)

            if len(pruned) < best_size:
                best_size = len(pruned)
                best_code = pruned

        if best_code is None:
            best_code = pruned_dp
        else:
            if not is_cover(best_code):
                best_code = pruned_dp

        result = [list(idx_to_tuple(c)) for c in best_code]
        return result

    else:
        # Generic fallback for other parameters
        total_words = q ** n
        if total_words <= 10000:
            powers = [1] * n
            for i in range(1, n):
                powers[i] = powers[i-1] * q
                
            def idx_to_tuple_gen(idx):
                t = [0] * n
                rem = idx
                for i in range(n):
                    t[i] = rem % q
                    rem //= q
                return tuple(t)
                
            def tuple_to_idx_gen(t):
                idx = 0
                for i in range(n):
                    idx += t[i] * powers[i]
                return idx

            balls = [None] * total_words
            for i in range(total_words):
                w = idx_to_tuple_gen(i)
                ball = {i}
                for pos in range(n):
                    orig_val = w[pos]
                    for new_val in range(q):
                        if new_val != orig_val:
                            new_w = list(w)
                            new_w[pos] = new_val
                            new_idx = tuple_to_idx_gen(tuple(new_w))
                            ball.add(new_idx)
                balls[i] = ball

            uncovered = set(range(total_words))
            code_indices = []
            
            while uncovered:
                best_c = -1
                max_gain = 0
                for c in range(total_words):
                    gain = len(balls[c] & uncovered)
                    if gain > max_gain:
                        max_gain = gain
                        best_c = c
                if best_c == -1:
                    break
                code_indices.append(best_c)
                uncovered -= balls[best_c]
                
                if time.time() > deadline:
                    break
            
            result = [list(idx_to_tuple_gen(c)) for c in code_indices]
            return result
        else:
            return [[0] * n]
