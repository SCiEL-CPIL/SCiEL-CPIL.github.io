import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Fallback for non-target parameters
    if not (q == 3 and n == 8 and radius == 1):
        # Use the seed logic for other cases
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Handle small n generically if needed, though task is fixed
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    N = q ** n
    start_time = time.time()
    time_limit = 120.0
    safety_margin = 5.0
    
    # Precompute balls (coverage sets)
    # balls[i] is a list of indices j covered by codeword i
    balls = []
    # M is the incidence matrix
    M = np.zeros((N, N), dtype=np.uint8)
    
    # Helper to convert int to base-q digits
    def int_to_digits(num, length):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % q
            num //= q
        return digits

    # Helper to convert base-q digits to int
    def digits_to_int(digits):
        val = 0
        for i in range(len(digits) - 1, -1, -1):
            val = val * q + digits[i]
        return val

    for i in range(N):
        digits = int_to_digits(i, n)
        ball = [i]
        M[i, i] = 1
        
        for k in range(n):
            orig_val = digits[k]
            for v in range(q):
                if v == orig_val:
                    continue
                new_digits = digits.copy()
                new_digits[k] = v
                j = digits_to_int(new_digits)
                if j not in ball: # Avoid duplicates if any (though logic prevents it)
                    ball.append(j)
                    M[i, j] = 1
        balls.append(ball)

    # Phase 1: Greedy Construction
    uncovered = np.ones(N, dtype=np.uint8)
    codewords = []
    
    while uncovered.any():
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        # Calculate how many uncovered words each candidate covers
        # M is N x N, uncovered is N x 1 (implicitly)
        scores = M.dot(uncovered)
        
        best_idx = np.argmax(scores)
        
        if scores[best_idx] == 0:
            # Should not happen if uncovered is not empty
            break
            
        codewords.append(best_idx)
        # Update uncovered
        # M[best_idx] is a row of 0s and 1s
        # We want to set uncovered[j] = 0 where M[best_idx, j] = 1
        uncovered[M[best_idx] == 1] = 0

    # Phase 2: Greedy Removal
    # Initialize counts
    counts = np.zeros(N, dtype=np.int32)
    for c in codewords:
        counts[balls[c]] += 1

    changed = True
    while changed and (time.time() - start_time < time_limit - safety_margin):
        changed = False
        new_codewords = []
        for c in codewords:
            mask = balls[c]
            counts[mask] -= 1
            if np.min(counts) >= 1:
                # We can remove c
                changed = True
            else:
                # We must keep c
                counts[mask] += 1
                new_codewords.append(c)
        codewords = new_codewords
        
    # Phase 3: Local Search (Swap Optimization)
    # Try to swap existing codewords with non-codewords to reduce size or improve structure
    # We can only swap if we can remove one and add one? No, we want to reduce size.
    # Actually, we can try to replace a codeword with another codeword that is not in the set,
    # if the new set still covers everything.
    # Or better: Try to find a codeword that can be removed, and then see if we can add a "better" one?
    # Since we already did greedy removal, we are in a minimal cover (no single word removable).
    # To get smaller, we need to change the set.
    # Strategy: Pick a codeword c, remove it, and add a new word c' that covers the holes,
    # such that c' is "better" (e.g., allows further removals later).
    # Or simply: Randomized restarts of the removal phase?
    # Or: Swap-based local search.
    # Let's try: For each c in codewords, try to replace it with c' not in codewords.
    # Condition: The set (codewords - {c} + {c'}) must be a valid cover.
    # This doesn't reduce size, but might lead to a state where greedy removal works better.
    
    # A more direct approach to reduce size:
    # We know K_3(8,1) <= 486.
    # Current greedy might be > 486.
    # Let's try to add a "smarter" removal or swap.
    
    # Actually, a known technique for covering codes is "pairwise swap" or "local search".
    # Let's implement a simple local search:
    # 1. Identify all words NOT in codewords.
    # 2. For each c in codewords, and each c' not in codewords (sampled or limited):
    #    Check if (C - {c} + {c'}) is a cover.
    #    If yes, accept the swap.
    #    This maintains size, but explores the space.
    #    After a swap, run greedy removal again.
    
    # Given time constraints, we might not be able to do exhaustive swaps.
    # Let's try a few random swaps followed by removal.
    
    current_size = len(codewords)
    best_codewords = codewords[:]
    best_size = current_size
    
    # We need a fast way to check if a set is a cover.
    # We can maintain the `counts` array.
    # If we swap c -> c':
    # counts[balls[c]] -= 1
    # counts[balls[c']] += 1
    # If min(counts) >= 1, it's a valid cover.
    
    # We also need to track which indices are in codewords to avoid adding duplicates.
    in_code = np.zeros(N, dtype=np.uint8)
    for c in codewords:
        in_code[c] = 1
        
    # Update counts to reflect current codewords
    counts = np.zeros(N, dtype=np.int32)
    for c in codewords:
        counts[balls[c]] += 1
        
    # Perform local search
    # We have a time budget. Let's use the remaining time.
    # Each swap check is O(1) if we maintain counts, but we need to iterate candidates.
    # Number of candidates is N - current_size. N=6561.
    # Iterating all candidates for each c is 6561 * 6561 ~ 43M ops. Too slow in Python loop.
    # We can vectorize the check for a fixed c.
    
    # Let's try to find a swap that allows removal.
    # Or just a valid swap that leads to a "better" state.
    # Since we want to minimize size, a valid swap is only useful if it enables removal.
    # So:
    # 1. Try swap c -> c'.
    # 2. If valid, run greedy removal.
    # 3. If size decreases, keep it. Else, revert.
    
    # To speed this up, we can prioritize c's that are "redundant" in some sense?
    # Or just iterate.
    
    # Let's try a limited number of iterations.
    max_swaps = 100
    for _ in range(max_swaps):
        if time.time() - start_time > time_limit - safety_margin:
            break
        if len(codewords) == 0:
            break
            
        # Pick a random codeword to try swapping out
        c_idx = random.randint(0, len(codewords) - 1)
        c = codewords[c_idx]
        
        # Remove c temporarily
        in_code[c] = 0
        counts[balls[c]] -= 1
        
        # Find a candidate c' to add
        # We want c' such that min(counts + balls[c']) >= 1
        # i.e., counts[j] + balls[c'][j] >= 1 for all j
        # i.e., wherever counts[j] == 0, balls[c'][j] must be 1.
        # Let zero_mask = (counts == 0)
        # We need balls[c'] to cover all indices in zero_mask.
        
        zero_mask = (counts == 0)
        if not zero_mask.any():
            # We can remove c entirely!
            # This shouldn't happen if we did greedy removal, but maybe after a swap?
            # If we just removed c and counts are all >= 1, then c was redundant.
            # Remove it from codewords list.
            codewords.pop(c_idx)
            # Do not add c'
            # Re-run greedy removal?
            # Yes, because removing c might make others redundant.
            # But we are in a loop. Let's just break and let the outer loop handle it?
            # Or just continue to next swap attempt.
            # Actually, if we removed c, we should probably run greedy removal again.
            # But for now, let's just accept the removal and continue.
            # We need to be careful with indices if we pop.
            # Let's restart the removal phase if this happens.
            # For simplicity, let's just break out of swap loop and go to removal.
            break
            
        # We need to find c' not in in_code such that balls[c'] covers zero_mask.
        # This is a set cover problem on zero_mask with candidates balls[c'].
        # We can check candidates one by one or vectorize.
        
        # Vectorized check:
        # We have M (N x N).
        # We want to find row c' such that M[c', zero_mask] are all 1.
        # And in_code[c'] == 0.
        
        # Let's compute scores for all candidates.
        # score[c'] = sum(M[c', zero_mask])
        # We want score[c'] == sum(zero_mask) (full coverage of holes)
        
        # M is N x N. zero_mask is N x 1.
        # We can compute M[:, zero_mask].sum(axis=1)
        
        # But M is huge (6561x6561).
        # Let's just iterate a few random candidates to save time.
        found_swap = False
        # Try 100 random candidates
        for _ in range(100):
            c_prime = random.randint(0, N - 1)
            if in_code[c_prime]:
                continue
                
            # Check if c_prime covers all holes
            # holes = np.where(zero_mask)[0]
            # if np.all(M[c_prime, holes]):
            #    swap
            
            # Faster:
            # if M[c_prime, zero_mask].all():
            
            # Let's use the precomputed balls list for speed?
            # balls[c_prime] is a list of indices.
            # We need to check if all indices in zero_mask are in balls[c_prime].
            # This is slow in Python.
            
            # Let's use M.
            # M is uint8.
            # M[c_prime, zero_mask] returns an array of 0s and 1s.
            # .all() checks if all are 1.
            
            if M[c_prime, zero_mask].all():
                # Valid swap
                in_code[c_prime] = 1
                counts[balls[c_prime]] += 1
                codewords[c_idx] = c_prime
                found_swap = True
                break
                
        if not found_swap:
            # Revert removal of c
            in_code[c] = 1
            counts[balls[c]] += 1
            
    # After swaps, run greedy removal again to clean up
    if time.time() - start_time < time_limit - safety_margin:
        # Re-initialize counts from current codewords
        counts = np.zeros(N, dtype=np.int32)
        for c in codewords:
            counts[balls[c]] += 1
            
        changed = True
        while changed and (time.time() - start_time < time_limit - safety_margin):
            changed = False
            new_codewords = []
            for c in codewords:
                mask = balls[c]
                counts[mask] -= 1
                if np.min(counts) >= 1:
                    changed = True
                else:
                    counts[mask] += 1
                    new_codewords.append(c)
            codewords = new_codewords

    # Convert back to lists of integers
    result = []
    for c in codewords:
        digits = int_to_digits(c, n)
        # The problem asks for "list of n integers in 0..q-1"
        # Order of digits: int_to_digits produces least significant first.
        # Usually "word" implies standard order.
        # The seed program: [a, b, ...]
        # itertools.product generates (0,0...), (0,1...), etc.
        # My int_to_digits(1) -> [1, 0, 0...] (LSB first).
        # I should reverse it to match standard MSB-first representation if that's what's expected?
        # The evaluator checks Hamming distance. It likely treats the list as a vector.
        # As long as the mapping is consistent, it doesn't matter for validity.
        # But to be safe and match "list of n integers", I'll just output the digits.
        # Let's check the seed:
        # tetracode = [[a, b, ...]]
        # tail = list(itertools.product(...))
        # The seed output order is MSB-like (first element is first coordinate).
        # My int_to_digits gives LSB first.
        # I will reverse the list to be MSB first.
        result.append(digits[::-1])

    return result
