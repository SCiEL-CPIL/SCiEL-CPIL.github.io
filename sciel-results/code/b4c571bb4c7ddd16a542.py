import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # Total words
    total_words = q ** n
    
    # Precompute neighbors and masks
    # word_to_idx and idx_to_word
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

    balls = []
    ball_masks = []
    
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors = [idx]
        for pos in range(n):
            original = w[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                w[pos] = new_val
                n_idx = word_to_idx(w)
                neighbors.append(n_idx)
            w[pos] = original
        
        balls.append(neighbors)
        mask = 0
        for nb in neighbors:
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
        Local search with 1-swap and 2-1 swap.
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
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to remove redundant codewords
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                redundant = True
                for w in balls[c]:
                    if cover_count[w] == 1:
                        redundant = False
                        break
                if redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    in_code[c] = False
                    code.pop(i)
                    improved = True
            
            if improved:
                continue
                
            # 2. Try 1-swap: remove c, add c' if it covers loss of c
            # and if this allows a reduction?
            # Actually, just do the swap if it maintains coverage.
            # Then check for reduction.
            # To be efficient, only swap if loss(c) is covered by c'.
            
            swapped = False
            for i in range(len(code)):
                c = code[i]
                # Calculate loss mask
                loss_mask = 0
                for w in balls[c]:
                    if cover_count[w] == 1:
                        loss_mask |= (1 << w)
                        
                if loss_mask == 0:
                    continue # Should be handled by reduction, but safe check
                    
                # Find c' that covers loss_mask
                # We want to find a c' that is NOT in code
                # And ideally, c' should overlap with other codewords to allow reduction?
                # For now, just find ANY c'.
                # Optimization: check c' that have high overlap with loss?
                
                best_c_prime = -1
                # We can break early if we find one?
                # Or pick one that maximizes "new coverage" or "overlap with other losses"?
                # Let's just pick the first one found to keep it simple and fast.
                # But randomization might help escape local minima.
                
                candidates = []
                for c_prime in range(total_words):
                    if in_code[c_prime]:
                        continue
                    if (ball_masks[c_prime] & loss_mask) == loss_mask:
                        candidates.append(c_prime)
                        
                if candidates:
                    # Pick one randomly to diversify
                    c_prime = random.choice(candidates)
                    
                    # Perform swap
                    for w in balls[c]:
                        cover_count[w] -= 1
                    in_code[c] = False
                    code[i] = c_prime
                    
                    for w in balls[c_prime]:
                        cover_count[w] += 1
                    in_code[c_prime] = True
                    
                    swapped = True
                    break # Restart loop to check for reductions
            
            if swapped:
                improved = True
                continue
                
            # 3. Try 2-1 swap: remove c1, c2, add c'
            # Loss is L1 union L2
            # If c' covers L1 union L2, we reduce size by 1.
            
            reduced = False
            # To speed up, we can limit the number of pairs we check per iteration
            # or check pairs in a random order.
            
            # Pre-compute loss masks for all current codewords
            loss_masks = [0] * len(code)
            for i in range(len(code)):
                c = code[i]
                lm = 0
                for w in balls[c]:
                    if cover_count[w] == 1:
                        lm |= (1 << w)
                loss_masks[i] = lm
            
            # Try random pairs
            # Number of pairs is ~ 190^2 / 2 = 18000
            # Checking all pairs * all c' is 18000 * 2187 = 39M ops.
            # This is too slow for Python in a tight loop if we do it every iteration.
            # We should only do this if 1-swap didn't work, and maybe limit it.
            
            if len(code) > 100:
                # Sample some pairs
                num_pairs_to_check = 500
                pairs_checked = 0
                while pairs_checked < num_pairs_to_check and time.time() < deadline:
                    i1 = random.randint(0, len(code) - 1)
                    i2 = random.randint(0, len(code) - 1)
                    if i1 == i2:
                        continue
                    pairs_checked += 1
                    
                    c1 = code[i1]
                    c2 = code[i2]
                    
                    # Union of losses
                    # Words covered by ONLY c1 or ONLY c2
                    # Note: if a word is covered by c1, c2, and c3, it is not in loss(c1) nor loss(c2).
                    # If we remove c1 and c2, words in loss(c1) \ loss(c2) and loss(c2) \ loss(c1) become uncovered.
                    # Actually, loss(c1) is words where count==1 and c1 is the one.
                    # So loss(c1) and loss(c2) are disjoint sets of words?
                    # Yes, because if a word w has count 1, it is covered by exactly one codeword.
                    # So loss(c1) and loss(c2) are disjoint.
                    
                    combined_loss = loss_masks[i1] | loss_masks[i2]
                    
                    if combined_loss == 0:
                        continue # Both redundant? Should have been removed.
                    
                    # Find c' that covers combined_loss
                    found = False
                    # Check all c'? Or sample?
                    # Let's check all c' but break early if found?
                    # We want to find ONE.
                    
                    for c_prime in range(total_words):
                        if in_code[c_prime]:
                            continue
                        if (ball_masks[c_prime] & combined_loss) == combined_loss:
                            # Perform swap
                            # Remove c1
                            for w in balls[c1]:
                                cover_count[w] -= 1
                            in_code[c1] = False
                            # Remove c2
                            for w in balls[c2]:
                                cover_count[w] -= 1
                            in_code[c2] = False
                            
                            # Remove from code list
                            # Careful with indices
                            # Remove the larger index first
                            if i1 > i2:
                                code.pop(i1)
                                code.pop(i2)
                            else:
                                code.pop(i2)
                                code.pop(i1)
                                
                            # Add c_prime
                            code.append(c_prime)
                            for w in balls[c_prime]:
                                cover_count[w] += 1
                            in_code[c_prime] = True
                            
                            found = True
                            break
                            
                    if found:
                        reduced = True
                        break
            
            if reduced:
                improved = True
                continue
                
            # If no improvement found in this pass, break
            # To avoid getting stuck, we could randomize the order of c or c'
            # But for now, if we can't improve, we stop this local search instance.
            
        return code

    # Main loop
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
        # Fallback: Direct sum with tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = []
    for idx in best_solution:
        result.append(idx_to_word(idx))
        
    return result
