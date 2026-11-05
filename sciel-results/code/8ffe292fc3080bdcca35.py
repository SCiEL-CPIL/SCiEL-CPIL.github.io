import time
import random
import numpy as np
import itertools

def _precompute_balls(q, n, radius):
    total = q ** n
    balls = [None] * total
    # Precompute ball for each word
    for idx in range(total):
        word = np.unravel_index(idx, (q,) * n)
        ball = np.zeros(total, dtype=bool)
        ball[idx] = True
        if radius >= 1:
            for pos in range(n):
                orig_val = word[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    new_word = list(word)
                    new_word[pos] = val
                    new_idx = np.ravel_multi_index(tuple(new_word), (q,) * n)
                    ball[new_idx] = True
        balls[idx] = ball
    return balls

def _get_candidates_for_set(S, balls, point_to_words):
    """
    Returns a list of word indices w such that S is a subset of balls[w].
    S is a set of indices.
    point_to_words[p] is a set of word indices w such that p in balls[w].
    """
    if not S:
        return list(range(len(balls))) # Any word works
    candidates = None
    for p in S:
        if candidates is None:
            candidates = set(point_to_words[p])
        else:
            candidates &= point_to_words[p]
            if not candidates:
                return []
    return list(candidates)

def _compute_exclusive_sets(code, balls):
    """
    Returns a list S_i where S_i is the set of points covered ONLY by code[i].
    """
    n_code = len(code)
    if n_code == 0:
        return []
    
    # Union of all balls
    U = np.zeros(len(balls[0]), dtype=bool)
    for i in range(n_code):
        U |= balls[code[i]]
        
    exclusive = []
    for i in range(n_code):
        # Points covered by i but not by any other
        # S_i = Ball(i) & ~(Union of all other balls)
        # Union of all other balls = U & ~Ball(i) ? No.
        # Let U_other = Union_{j != i} Ball(j)
        # We can compute U_other as U & ~Ball(i) IF Ball(i) is a subset of U. Which it is.
        # Wait, U = Ball(i) | U_other.
        # So U & ~Ball(i) = U_other.
        # Then S_i = Ball(i) & ~U_other = Ball(i) & ~(U & ~Ball(i))
        # This simplifies to Ball(i) & ~U_other.
        # But we can just do:
        # U_other = np.copy(U)
        # U_other[balls[code[i]]] = False # This is wrong, U is bool array.
        
        # Efficient way:
        # U_other = U & ~balls[code[i]]
        # S_i = balls[code[i]] & ~U_other
        # S_i = balls[code[i]] & ~(U & ~balls[code[i]])
        
        # Let's just compute U_other directly? That's O(N^2).
        # For N ~ 73, N^2 is ~5000. Each op is on 729 bools. 5000 * 729 ~ 3.6M ops. Very fast.
        
        U_other = np.zeros(len(balls[0]), dtype=bool)
        for j in range(n_code):
            if j != i:
                U_other |= balls[code[j]]
        
        S_i = balls[code[i]] & ~U_other
        exclusive.append(S_i)
    return exclusive

def _total_overlap(code, balls):
    """
    Calculate sum of overlaps to use as a secondary objective.
    """
    # Count how many codewords cover each point
    counts = np.zeros(len(balls[0]), dtype=np.int32)
    for cw in code:
        counts[balls[cw]] += 1
    # Sum of C(counts[p], 2)
    # Or just sum(counts) - n_code is a measure of redundancy
    return np.sum(counts)

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters (though task specifies 3,6,1)
        # Simple greedy
        total = q ** n
        balls = _precompute_balls(q, n, radius)
        uncovered = set(range(total))
        code = []
        while uncovered:
            best_idx = -1
            best_count = 0
            for idx in range(total):
                count = sum(1 for p in balls[idx] if p in uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = idx
            if best_count == 0:
                break
            code.append(best_idx)
            uncovered -= set(np.where(balls[best_idx])[0])
        return [list(np.unravel_index(cw, (q,)*n)) for cw in code]

    total = 729
    balls = _precompute_balls(q, n, radius)
    
    # Precompute point_to_words for faster candidate search
    point_to_words = [set() for _ in range(total)]
    for w in range(total):
        for p in np.where(balls[w])[0]:
            point_to_words[p].add(w)
    # Convert to lists for faster iteration if needed, or keep as sets for intersection
    # Sets are good for intersection.

    # Initialize with a random valid code
    # Start with empty, add random words until covered
    code = []
    U = np.zeros(total, dtype=bool)
    
    # Random initialization
    random.seed(42)
    while not U.all() and time.time() < deadline - 5:
        # Pick a random word
        # To be faster, pick a word that covers an uncovered point
        uncovered_indices = np.where(~U)[0]
        if len(uncovered_indices) == 0:
            break
        # Pick a random uncovered point
        p = uncovered_indices[random.randint(0, len(uncovered_indices)-1)]
        # Pick a random word covering p
        candidates = list(point_to_words[p])
        w = candidates[random.randint(0, len(candidates)-1)]
        if w not in code:
            code.append(w)
            U |= balls[w]
            
    if not U.all():
        # Fallback: greedy
        while not U.all():
            uncovered_indices = np.where(~U)[0]
            if len(uncovered_indices) == 0:
                break
            p = uncovered_indices[0]
            best_w = -1
            best_cov = -1
            for w in point_to_words[p]:
                if w in code: continue
                cov = np.sum(balls[w] & ~U)
                if cov > best_cov:
                    best_cov = cov
                    best_w = w
            if best_w == -1: break
            code.append(best_w)
            U |= balls[best_w]

    best_code = code.copy()
    best_len = len(code)
    
    # Local Search
    while time.time() < deadline - 2:
        # 1. Try to remove redundant codewords
        removed = False
        for i in range(len(code)):
            if time.time() > deadline - 2:
                break
            # Check if removing code[i] keeps coverage
            # U_new = U & ~balls[code[i]]
            # If U_new is all True, we can remove
            # Actually, we need to check if U \ Ball(i) is full.
            # Since U is full, U \ Ball(i) is full iff Ball(i) is subset of Union_{j!=i} Ball(j)
            # Which is equivalent to S_i being empty.
            
            # Compute S_i
            U_other = np.zeros(total, dtype=bool)
            for j in range(len(code)):
                if j != i:
                    U_other |= balls[code[j]]
            
            S_i = balls[code[i]] & ~U_other
            if not S_i.any():
                # Can remove
                code.pop(i)
                U = U_other # Update U
                removed = True
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = code.copy()
                break # Restart loop from 0 as indices changed
        
        if removed:
            continue
            
        # 2. Try 1-1 Swaps
        swapped = False
        for i in range(len(code)):
            if time.time() > deadline - 2:
                break
            
            # Compute S_i
            U_other = np.zeros(total, dtype=bool)
            for j in range(len(code)):
                if j != i:
                    U_other |= balls[code[j]]
            
            S_i = balls[code[i]] & ~U_other
            
            if S_i.any():
                # Find candidates w such that S_i is subset of balls[w]
                # And w not in code
                candidates = _get_candidates_for_set(set(np.where(S_i)[0]), balls, point_to_words)
                # Filter out current code
                code_set = set(code)
                candidates = [w for w in candidates if w not in code_set]
                
                if candidates:
                    # Heuristic: pick candidate that minimizes overlap with U_other
                    # Overlap = sum(balls[w] & U_other)
                    best_w = -1
                    best_overlap = float('inf')
                    # Sample candidates if too many
                    if len(candidates) > 50:
                        candidates = random.sample(candidates, 50)
                    
                    for w in candidates:
                        ov = np.sum(balls[w] & U_other)
                        if ov < best_overlap:
                            best_overlap = ov
                            best_w = w
                    
                    if best_w != -1:
                        # Perform swap
                        code[i] = best_w
                        # Update U
                        # U = U_other | balls[best_w]
                        U = U_other | balls[best_w]
                        swapped = True
                        # We don't update best_len here unless we remove later
                        break # Restart loop
        
        if swapped:
            continue
            
        # 3. Try 2-2 Swaps (if 1-1 stuck)
        # This is more expensive. Only do if time allows.
        if len(code) > 73 and time.time() < deadline - 10:
             # Pick two random indices
             if len(code) >= 2:
                 i, j = random.sample(range(len(code)), 2)
                 
                 # Compute S_ij: points covered ONLY by i or j
                 # U_other = Union of all k != i, j
                 U_other = np.zeros(total, dtype=bool)
                 for k in range(len(code)):
                     if k != i and k != j:
                         U_other |= balls[code[k]]
                 
                 # Points covered by i or j
                 U_ij = balls[code[i]] | balls[code[j]]
                 # Exclusive to pair
                 S_ij = U_ij & ~U_other
                 
                 if S_ij.any():
                     # We need w1, w2 such that S_ij is subset of balls[w1] | balls[w2]
                     # And w1, w2 not in code
                     code_set = set(code)
                     
                     # Strategy:
                     # 1. Find candidates for w1 that cover a large part of S_ij
                     # 2. Find candidates for w2 that cover the rest
                     
                     # Simple approach: Random search for w1, then w2
                     found = False
                     for _ in range(20): # Limit iterations
                         if time.time() > deadline - 2:
                             break
                         # Pick w1 from all words not in code
                         # Prefer words that cover many points in S_ij
                         all_words = [w for w in range(total) if w not in code_set]
                         if not all_words:
                             break
                         # Sample
                         sample = random.sample(all_words, min(50, len(all_words)))
                         # Sort by coverage of S_ij
                         sample.sort(key=lambda w: np.sum(balls[w] & S_ij), reverse=True)
                         
                         for w1 in sample[:5]:
                             # Remaining points to cover by w2
                             rem = S_ij & ~balls[w1]
                             if not rem.any():
                                 # w1 covers all. w2 can be any word not in code
                                 # But we are replacing 2 words with 2 words.
                                 # If w1 covers all S_ij, we can pick any w2 not in code.
                                 # But we must ensure the union of balls[w1] and balls[w2] covers S_ij.
                                 # It does.
                                 # We also need to ensure the new code is valid.
                                 # The rest of the space is covered by U_other.
                                 # So yes, valid.
                                 # Pick w2 randomly
                                 w2 = random.choice(all_words)
                                 if w2 == w1:
                                     # Need distinct codewords? Task says distinct.
                                     # If w1 == w2, we have duplicate.
                                     # Pick another
                                     if len(all_words) > 1:
                                         w2 = random.choice([w for w in all_words if w != w1])
                                     else:
                                         continue
                                 
                                 # Perform swap
                                 code[i] = w1
                                 code[j] = w2
                                 # Update U
                                 U = U_other | balls[w1] | balls[w2]
                                 found = True
                                 break
                             else:
                                 # Find w2 covering rem
                                 candidates_w2 = _get_candidates_for_set(set(np.where(rem)[0]), balls, point_to_words)
                                 candidates_w2 = [w for w in candidates_w2 if w not in code_set and w != w1]
                                 if candidates_w2:
                                     w2 = random.choice(candidates_w2)
                                     # Perform swap
                                     code[i] = w1
                                     code[j] = w2
                                     U = U_other | balls[w1] | balls[w2]
                                     found = True
                                     break
                         if found:
                             break
                     
                     if found:
                         continue # Restart main loop

    # Final cleanup: remove any remaining redundant codewords
    for i in range(len(code)):
        if time.time() > deadline - 1:
            break
        U_other = np.zeros(total, dtype=bool)
        for j in range(len(code)):
            if j != i:
                U_other |= balls[code[j]]
        S_i = balls[code[i]] & ~U_other
        if not S_i.any():
            code.pop(i)
            break # Restart

    # Return the best code found
    # The current 'code' might be better than 'best_code' if we didn't update best_code on swap
    # But we only update best_code on removal.
    # If we swapped, the length is the same.
    # We should return the shortest valid code.
    # Since we only perform valid swaps/removals, 'code' is always valid.
    # If len(code) < best_len, update best_code.
    if len(code) < best_len:
        best_code = code.copy()
        
    # Convert indices to lists of integers
    result = [list(np.unravel_index(cw, (q,)*n)) for cw in best_code]
    return result
