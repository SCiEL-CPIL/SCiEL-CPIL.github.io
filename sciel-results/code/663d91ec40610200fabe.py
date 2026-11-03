import numpy as np
import time
import sys

def construct(q, n, radius):
    """
    Constructs a covering code using a greedy set cover heuristic.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Parameters specific to the task: q=3, n=7, radius=1
    # But we write it generally enough to be robust if parameters vary slightly, 
    # though the task specifies the 3-7-1 case.
    
    total_words = q ** n
    
    # If the space is too large for memory or time, we might need a different strategy,
    # but for n=7, q=3, total_words = 2187, which is very small.
    # We can afford to precompute balls.
    
    # Precompute all possible words as integers
    # We use a lookup table for digits to speed up conversion if needed, 
    # but for n=7, direct iteration is fast enough.
    
    # Generate all words as integers 0 to q^n - 1
    # word i corresponds to the base-q representation of i.
    
    # Precompute the balls (sets of words covered by each potential codeword)
    # ball[c] is a boolean array or list of words covered by c
    # To save memory and time, we can store the list of indices.
    
    # Helper to convert int to base-q digits
    def int_to_base_q(x, n, q):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits

    # Helper to convert base-q digits to int
    def base_q_to_int(digits, q):
        x = 0
        for i in range(n):
            x = x * q + digits[i]
        return x

    # Precompute balls for all 2187 words
    # This is O(q^n * n * q) which is small.
    balls = [None] * total_words
    
    for c in range(total_words):
        covered = set()
        covered.add(c)
        digits = int_to_base_q(c, n, q)
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                digits[pos] = new_val
                w = base_q_to_int(digits, q)
                covered.add(w)
            digits[pos] = orig_val
        balls[c] = covered
        
    # Greedy Set Cover
    # uncovered: set of words not yet covered
    uncovered = set(range(total_words))
    codewords = []
    
    # We can speed this up by not checking all candidates if we are close to the end,
    # but with only 2187 candidates, checking all is fast.
    # However, the "covered" sets are large. 
    # Intersection size calculation: len(covered & uncovered)
    
    # To optimize, we can maintain a count of how many times each word is covered by *potential* codewords?
    # No, standard greedy is:
    # 1. Find c that maximizes |ball[c] \cap uncovered|
    # 2. Add c to codewords
    # 3. uncovered -= ball[c]
    
    # This is O(NumSteps * TotalWords * BallSize)
    # NumSteps ~ 200. TotalWords ~ 2200. BallSize ~ 21.
    # 200 * 2200 * 21 ~ 9 million ops. Very fast in Python? 
    # Set intersection in Python is C-optimized.
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_c = -1
        best_count = -1
        
        # Optimization: If we have many candidates, we might sample. 
        # But 2187 is small. We iterate all.
        for c in range(total_words):
            # If c is already a codeword, we could skip, but its ball is already subtracted.
            # Actually, we should skip c if it's already in codewords? 
            # No, we can just check the intersection.
            # But wait, if c is in codewords, ball[c] is a subset of (Total - Uncovered).
            # So intersection will be 0 or small.
            
            # A small optimization: if best_count is already high, we might prune?
            # Hard to prune without knowing the max possible.
            
            count = len(balls[c] & uncovered)
            if count > best_count:
                best_count = count
                best_c = c
                
        if best_c == -1 or best_count == 0:
            # Should not happen if uncovered is not empty
            break
            
        codewords.append(best_c)
        uncovered -= balls[best_c]
        
    # Convert integer codewords back to lists of digits
    result = []
    for c in codewords:
        result.append(int_to_base_q(c, n, q))
        
    return result

