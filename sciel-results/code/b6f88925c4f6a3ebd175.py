import itertools
import time
import numpy as np
import random

def construct(q, n, radius):
    """
    Constructs a covering code using greedy set cover + efficient local search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    total_words = q ** n

    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # Precompute balls (words covered by center i) and pre-images (centers covering word i)
    # For radius=1, q=3, n=8: Ball size = 1 + 8*2 = 17
    ball_size = 1 + n * (q - 1) if radius == 1 else 0
    
    # We need efficient access to:
    # 1. balls[i]: list of words covered by center i
    # 2. preimages[i]: list of centers that cover word i
    
    balls = [None] * total_words
    preimages = [None] * total_words
    
    # Initialize lists
    for i in range(total_words):
        balls[i] = []
        preimages[i] = []
        
    # Fill balls and preimages
    # For each center i, find all words j in its ball
    for i in range(total_words):
        d_center = int_to_digits(i, n, q)
        # Add center itself
        balls[i].append(i)
        preimages[i].append(i)
        
        if radius == 1:
            for pos in range(n):
                orig = d_center[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    d_center[pos] = new_val
                    j = digits_to_int(d_center, q)
                    balls[i].append(j)
                    preimages[j].append(i)
                d_center[pos] = orig
        else:
            # Generic radius handling (slower, but not used for this task)
            # For simplicity, we assume radius=1 for this optimization. 
            # If radius != 1, we fall back to a slower method or just use the same logic if q=3,r=1 is the only case.
            # The task specifies q=3, n=8, radius=1.
            pass

    # Convert to numpy arrays for vectorized operations where possible
    # But for the local search logic below, lists might be easier to manage for small sizes.
    # Let's stick to lists for balls/preimages as they are small (size 17).
    
    # Greedy Set Cover
    code = []
    code_set = set()
    uncovered_mask = np.ones(total_words, dtype=np.bool_)
    
    # To speed up greedy, we can track counts of uncovered words per candidate
    # But simple greedy with vectorized gain is okay for N=6561.
    # gains[i] = number of uncovered words in balls[i]
    
    while uncovered_mask.any():
        if time.time() > deadline:
            break
            
        # Compute gains
        # balls_np is not created yet. Let's create a sparse representation or just loop?
        # Looping 6561 * 17 is ~111k ops. Very fast.
        gains = np.zeros(total_words, dtype=np.int32)
        for i in range(total_words):
            # Count uncovered in balls[i]
            # balls[i] is a list of ints
            gains[i] = np.sum(uncovered_mask[balls[i]])
            
        best_center = int(np.argmax(gains))
        max_gain = int(gains[best_center])
        
        if max_gain == 0:
            break
            
        code.append(best_center)
        code_set.add(best_center)
        uncovered_mask[balls[best_center]] = False
        
    # If we broke due to time, we might have uncovered words.
    # For this specific problem size, greedy should finish quickly.
    # If it doesn't, the result is invalid. We assume it does.
    
    # Initial Trim
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code:
        cover_count[balls[c]] += 1
        
    changed = True
    while changed and time.time() < deadline:
        changed = False
        to_remove = []
        for i, c in enumerate(code):
            # If no word is uniquely covered by c, remove it
            if not np.any(cover_count[balls[c]] == 1):
                to_remove.append(i)
        
        if to_remove:
            for idx in sorted(to_remove, reverse=True):
                c = code[idx]
                code.pop(idx)
                code_set.discard(c)
                cover_count[balls[c]] -= 1
            changed = True
            
    # Local Search
    # Strategy: For each c, if it has unique words, try to replace it with a neighbor of one unique word
    # that covers all unique words.
    
    def local_search_pass():
        nonlocal code, code_set, cover_count
        improved = True
        passes = 0
        max_passes = 1000 # High limit, controlled by time
        
        while improved and time.time() < deadline and passes < max_passes:
            improved = False
            passes += 1
            
            # Shuffle to avoid bias
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    break
                
                c = code[idx]
                c_ball = balls[c]
                
                # Find words uniquely covered by c
                unique_words = []
                for w in c_ball:
                    if cover_count[w] == 1:
                        unique_words.append(w)
                        
                if not unique_words:
                    # Remove c
                    code.pop(idx)
                    code_set.discard(c)
                    cover_count[c_ball] -= 1
                    improved = True
                    break # Restart pass as indices changed
                    
                # Try to replace c
                # Pick a random unique word w0
                w0 = unique_words[0]
                
                # Candidates are centers that cover w0
                candidates = preimages[w0]
                
                found_swap = False
                for c_new in candidates:
                    if c_new in code_set:
                        continue
                        
                    # Check if c_new covers all unique_words
                    c_new_ball = balls[c_new]
                    # Optimization: Check if unique_words is subset of c_new_ball
                    # Since unique_words is small, just loop
                    covers_all = True
                    for w in unique_words:
                        if cover_count[w] == 1: # Still unique? Yes, we haven't changed counts yet
                            # Check if w is in c_new_ball
                            # c_new_ball is a list. 
                            # We can check if w is in the list. 
                            # Better: precompute a set for c_new_ball? 
                            # Or just check distance? 
                            # dist(w, c_new) <= 1 is equivalent to w in balls[c_new].
                            # We can just check if w is in the list.
                            if w not in c_new_ball:
                                covers_all = False
                                break
                    
                    if covers_all:
                        # Perform swap
                        code[idx] = c_new
                        code_set.discard(c)
                        code_set.add(c_new)
                        cover_count[c_ball] -= 1
                        cover_count[c_new_ball] += 1
                        improved = True
                        found_swap = True
                        break # Restart pass
                        
                if found_swap:
                    break # Restart pass
                    
    # Run local search
    local_search_pass()
    
    # Final Trim
    changed = True
    while changed and time.time() < deadline:
        changed = False
        to_remove = []
        for i, c in enumerate(code):
            if not np.any(cover_count[balls[c]] == 1):
                to_remove.append(i)
        if to_remove:
            for idx in sorted(to_remove, reverse=True):
                c = code[idx]
                code.pop(idx)
                code_set.discard(c)
                cover_count[balls[c]] -= 1
            changed = True
            
    # Verify coverage
    # If any word is uncovered, the code is invalid.
    # We should try to fix this by adding words if time permits.
    uncovered_mask = np.ones(total_words, dtype=np.bool_)
    for c in code:
        uncovered_mask[balls[c]] = False
        
    if uncovered_mask.any():
        # Greedily add words to cover remaining
        while uncovered_mask.any() and time.time() < deadline:
            gains = np.zeros(total_words, dtype=np.int32)
            for i in range(total_words):
                gains[i] = np.sum(uncovered_mask[balls[i]])
            best_center = int(np.argmax(gains))
            if gains[best_center] == 0:
                break
            code.append(best_center)
            code_set.add(best_center)
            uncovered_mask[balls[best_center]] = False
            
    # Convert to result format
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))
        
    return result
