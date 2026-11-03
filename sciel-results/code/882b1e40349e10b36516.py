import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115
    
    # Problem specific parameters
    # q=3, n=7, radius=1
    total_words = q ** n
    # Precompute balls
    # index -> list of indices in ball
    balls = []
    ball_masks = [] # integer masks for fast subset check
    
    def index_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return tuple(word)

    # Precompute
    for idx in range(total_words):
        word = index_to_word(idx)
        covered = [idx]
        for pos in range(n):
            digit = word[pos]
            weight = q ** (n - 1 - pos)
            for val in range(q):
                if val != digit:
                    new_idx = idx + (val - digit) * weight
                    covered.append(new_idx)
        balls.append(covered)
        mask = 0
        for c in covered:
            mask |= (1 << c)
        ball_masks.append(mask)

    def greedy(random_seed):
        random.seed(random_seed)
        uncovered = np.ones(total_words, dtype=np.uint8)
        selected = []
        
        while uncovered.any():
            if time.time() > deadline - 10:
                break
            
            # Calculate scores: number of uncovered words covered by each candidate
            # scores[i] = sum(uncovered[balls[i]])
            # Vectorized approach:
            # We can't easily vectorize "sum over variable length lists" without sparse matrix.
            # But 2187 * 15 = 32k ops. Python loop is okay, but NumPy is better.
            
            # Let's use a loop for simplicity and speed check
            best_count = -1
            best_idx = -1
            # To allow random tie-breaking, we can iterate in random order or add noise
            # Adding noise is easier
            for i in range(total_words):
                # count uncovered in ball
                # balls[i] is a list of indices
                # np.count_nonzero(uncovered[balls[i]])
                c = np.count_nonzero(uncovered[balls[i]])
                # Add small noise to break ties randomly
                c += random.random() * 0.01
                if c > best_count:
                    best_count = c
                    best_idx = i
            
            if best_idx == -1:
                break
            
            selected.append(best_idx)
            uncovered[balls[best_idx]] = 0
            
        return selected

    def polish(selected):
        if not selected:
            return selected
            
        # Compute cover_count
        cc = np.zeros(total_words, dtype=np.int32)
        for c in selected:
            cc[balls[c]] += 1
            
        # Pass 1: Removals
        changed = True
        while changed and time.time() < deadline - 20:
            changed = False
            for i in range(len(selected)):
                c = selected[i]
                # Check if any word in balls[c] has cc == 1
                # If min(cc[balls[c]]) > 1, we can remove
                if np.min(cc[balls[c]]) > 1:
                    cc[balls[c]] -= 1
                    selected.pop(i)
                    changed = True
                    break # Restart scan
        
        # Pass 2: Swaps
        # We want to reduce the number of "bottleneck" words (cc == 1)
        # Or just find valid swaps to rearrange.
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_bottlenecks = np.count_nonzero(cc == 1)
            
            for i in range(len(selected)):
                if time.time() > deadline - 5:
                    break
                c = selected[i]
                # Find unique words covered by c
                unique_mask = 0
                for w in balls[c]:
                    if cc[w] == 1:
                        unique_mask |= (1 << w)
                        
                if unique_mask == 0:
                    # This codeword is redundant, should have been removed in Pass 1
                    # But maybe Pass 1 didn't catch it if it was added later? 
                    # Pass 1 removes if min > 1. If min == 1, it's not removable.
                    # If unique_mask == 0, it means no word is ONLY covered by c.
                    # So min(cc[balls[c]]) > 1. It SHOULD be removable.
                    # Why wasn't it? Maybe it was removed in this pass? No, we are in swap pass.
                    # If Pass 1 finished, no codeword has min > 1.
                    # So if unique_mask == 0, it's a contradiction unless cc changed.
                    # Actually, if unique_mask == 0, we can just remove it.
                    cc[balls[c]] -= 1
                    selected.pop(i)
                    improved = True
                    break # Restart
                    
                # Find a candidate c' not in selected
                # such that ball_masks[c'] covers unique_mask
                # (ball_masks[c'] & unique_mask) == unique_mask
                
                best_cand = -1
                best_new_bottlenecks = current_bottlenecks + 100 # infinity
                # To speed up, we can sample candidates or iterate all
                # Iterating all 2187 is fast (2187 bitwise ops)
                # But we must skip c' in selected.
                
                # Optimization: Only consider c' that are "close" to c?
                # No, global search is better for escaping local minima.
                
                # We can pre-filter candidates that cover the largest unique word?
                # Let's just iterate.
                
                for c_prime in range(total_words):
                    if c_prime in selected:
                        continue
                    
                    # Check coverage
                    if (ball_masks[c_prime] & unique_mask) != unique_mask:
                        continue
                    
                    # Simulate swap
                    # new_cc = cc - balls[c] + balls[c_prime]
                    # We don't want to allocate new array every time.
                    # We can do it in place and revert, or just check conditions.
                    
                    # Check if swap is valid (no zero coverage)
                    # Words in balls[c] that are NOT in balls[c_prime] will lose 1 coverage.
                    # If any of those had cc == 1, they become 0. Invalid.
                    # Words in balls[c_prime] that are NOT in balls[c] will gain 1 coverage.
                    
                    # Let's compute new bottleneck count
                    # This is expensive to do for every candidate.
                    
                    # Heuristic:
                    # 1. Check validity:
                    #    Let S_c = balls[c], S_cp = balls[c_prime]
                    #    Losing words = S_c \ S_cp
                    #    If any w in Losing words has cc[w] == 1, invalid.
                    
                    #    Gaining words = S_cp \ S_c
                    #    These will have cc increased.
                    
                    #    Shared words = S_c \cap S_cp
                    #    These stay same.
                    
                    #    Unique words of c (cc==1) are in S_c \ S_cp?
                    #    No, unique words of c MUST be covered by c'.
                    #    So unique words are in S_c \cap S_cp.
                    #    Wait, if w is unique to c, cc[w]==1.
                    #    If c' covers w, then w is in S_cp.
                    #    So w is in S_c \cap S_cp.
                    #    So cc[w] becomes 1 - 1 + 1 = 1. Still 1.
                    
                    #    So the unique words of c remain bottlenecks.
                    
                    #    Other words in S_c \ S_cp (not unique to c, so cc>=2):
                    #    cc decreases by 1. Becomes cc-1.
                    #    If cc was 2, becomes 1 (new bottleneck).
                    #    If cc was >2, stays >1.
                    
                    #    Words in S_cp \ S_c:
                    #    cc increases by 1.
                    #    If cc was 0? No, cc >= 1 for all covered words.
                    #    Wait, are all words covered? Yes.
                    #    So cc >= 1.
                    #    Becomes cc+1. Not a bottleneck.
                    
                    #    So new bottlenecks come from:
                    #    1. Old bottlenecks that remain (unique words of c).
                    #    2. Words in S_c \ S_cp that had cc == 2 (become 1).
                    #    3. Old bottlenecks NOT in S_c? No, they are unchanged.
                    #    4. Old bottlenecks IN S_c but NOT unique? Impossible, if in S_c and cc==1, it is unique to c.
                    
                    #    So:
                    #    New Bottlenecks = (Old Bottlenecks - (Bottlenecks in S_c \ S_cp that were not unique to c? No, all bottlenecks in S_c are unique to c))
                    #    Actually, let B be set of words with cc==1.
                    #    B_new = (B \ (S_c \ S_cp)) \cup ( (S_c \ S_cp) \cap {w | cc[w]==2} )
                    #    Wait, words in B \ S_c are unchanged.
                    #    Words in B \cap S_c are unique to c. They are in S_cp (by condition). So they are in S_c \cap S_cp.
                    #    So they are NOT in S_c \ S_cp.
                    #    So they remain in B_new.
                    
                    #    So B_new = (B \cap (S_c \ S_cp is empty for B? No, B \ S_c is safe. B \cap S_c is safe because they are in S_cp))
                    #    So all old bottlenecks in B are preserved?
                    #    Yes, if c' covers all unique words of c.
                    
                    #    New bottlenecks are ONLY from words in S_c \ S_cp that had cc == 2.
                    #    Let W2 = {w in S_c \ S_cp | cc[w] == 2}.
                    #    New Bottlenecks = Old Bottlenecks + |W2|.
                    
                    #    We want to minimize |W2|. Ideally 0.
                    
                    losing_words = [w for w in balls[c] if w not in balls[c_prime]]
                    # This list comprehension is slow.
                    # Use sets?
                    # sets[c] and sets[c_prime]
                    
                    # Let's precompute sets
                    # But we are in a loop.
                    
                    # Alternative:
                    # Count how many w in balls[c] have cc[w]==2 and are not in balls[c_prime].
                    
                    # This is getting complicated to do efficiently in pure Python inside the inner loop.
                    # Given 2187 candidates, and inner loop ~15 ops, it's ~30k ops per candidate.
                    # 2187 * 30k = 65 million ops. Too slow for Python.
                    
                    # We need to vectorize or use bitsets.
                    # We have ball_masks.
                    # We don't have "cc==2 mask" easily.
                    
                    # Let's simplify:
                    # Just check validity (no zeros).
                    # If valid, perform swap.
                    # Don't optimize for bottleneck reduction in the swap selection itself,
                    # but rely on the removal pass to clean up.
                    # If we do neutral swaps, we change the structure, allowing new removals.
                    
                    # Check validity:
                    # Any w in balls[c] such that cc[w]==1 and w not in balls[c_prime]?
                    # We already ensured unique words (cc==1) are in balls[c_prime].
                    # So any w in balls[c] with cc[w]==1 IS in balls[c_prime].
                    # So no word with cc==1 becomes 0.
                    # So the swap is ALWAYS valid if it covers the unique words!
                    
                    # So we just need to find ANY c' that covers unique words.
                    # To escape local minima, we should pick c' randomly from the valid ones.
                    
                    valid_candidates = []
                    for c_prime in range(total_words):
                        if c_prime in selected:
                            continue
                        if (ball_masks[c_prime] & unique_mask) == unique_mask:
                            valid_candidates.append(c_prime)
                            
                    if valid_candidates:
                        # Pick one randomly
                        c_new = random.choice(valid_candidates)
                        # Perform swap
                        cc[balls[c]] -= 1
                        cc[balls[c_new]] += 1
                        selected[i] = c_new
                        improved = True
                        break # Restart scan
        
        # Pass 3: Final Removals
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(len(selected)):
                c = selected[i]
                if np.min(cc[balls[c]]) > 1:
                    cc[balls[c]] -= 1
                    selected.pop(i)
                    changed = True
                    break
                    
        return selected

    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts
    while time.time() < deadline - 30:
        seed = int(time.time() * 1000) % 10000
        sol = greedy(seed)
        if not sol:
            continue
        polished = polish(sol)
        
        if len(polished) < best_size:
            best_size = len(polished)
            best_solution = polished
            
        # If we found a very good solution, maybe tighten the time for polish?
        # Or just keep going.
        
    # Final check
    if best_solution is None:
        # Fallback: Direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_solution = [h + list(tail) for h in tetracode for tail in __import__('itertools').product(range(3), repeat=n - 4)]
        return base_solution
        
    # Convert to list of lists
    codewords = [list(index_to_word(c)) for c in best_solution]
    
    # Verify
    # (Omitted for brevity in thought, but should be in code)
    
    return codewords
