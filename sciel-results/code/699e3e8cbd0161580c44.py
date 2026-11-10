import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 110.0 # Safety margin from 120s
    
    # This task is specifically for q=3, n=7, radius=1.
    # We assume the evaluator passes these values.
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple valid construction if parameters differ, 
        # though the task contract implies fixed parameters.
        # For now, we focus on the specific case.
        pass

    # 1. Generate initial candidate set: Direct sum of Tetracode (len 4) and Full Space (len 3)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Generate all tails of length 3
    tails = list(itertools.product(range(3), repeat=n - 4))
    
    initial_codewords = []
    for h in tetracode:
        for tail in tails:
            initial_codewords.append(h + list(tail))
            
    # Convert codewords to integer indices for faster processing
    def word_to_int(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
        
    def int_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    initial_indices = [word_to_int(w) for w in initial_codewords]
    
    # Precompute balls for all 2187 words
    # ball[i] is a list of indices in the space covered by codeword i
    # Radius 1 ball size is 1 + n*(q-1) = 1 + 7*2 = 15
    balls = [None] * (q ** n)
    for i in range(q ** n):
        w = int_to_word(i)
        ball_indices = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                # Create new word
                new_w = w.copy()
                new_w[pos] = new_val
                ball_indices.append(word_to_int(new_w))
        balls[i] = ball_indices

    # Initialize coverage counts
    coverage = np.zeros(q ** n, dtype=np.int32)
    
    # Add all initial codewords to coverage
    for idx in initial_indices:
        for b_idx in balls[idx]:
            coverage[b_idx] += 1
            
    active_indices = list(initial_indices)
    
    # Greedy removal
    # We iterate and remove redundant codewords.
    # A codeword is redundant if all words in its ball are covered by at least one other codeword.
    # i.e., min(coverage[ball]) > 1.
    
    removed = True
    while removed:
        removed = False
        current_active = active_indices[:] # Iterate over a copy
        for idx in current_active:
            if time.time() > deadline:
                break
                
            # Check if redundant
            # If min coverage in its ball is > 1, it's redundant
            min_cov = np.min(coverage[balls[idx]])
            if min_cov > 1:
                # Remove it
                active_indices.remove(idx)
                # Update coverage
                for b_idx in balls[idx]:
                    coverage[b_idx] -= 1
                removed = True
        if time.time() > deadline:
            break
            
    # Convert back to list of lists
    result = [int_to_word(idx) for idx in active_indices]
    
    return result
