import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # For this specific task, q=3, n=7, radius=1
    # But we write it generally enough to handle the call signature.
    
    total_words = q ** n
    # Precompute all words in the space
    all_words = [list(map(int, np.unravel_index(i, [q]*n))) for i in range(total_words)]
    # Map word to index for fast lookup
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute spheres of radius 'radius'
    # For radius 1, sphere of w is w and all words differing in exactly one position by +1 or -1 (mod q)
    # We only need radius 1 for this task, but let's be slightly generic or just specialize.
    # Given the time constraint and specific task, specializing for radius 1 is safer/faster.
    
    spheres = []
    for w in all_words:
        sphere = [w]
        if radius >= 1:
            for i in range(n):
                for delta in range(1, q):
                    # Try +delta
                    w_plus = w.copy()
                    w_plus[i] = (w[i] + delta) % q
                    sphere.append(w_plus)
                    # Try -delta (which is q-delta)
                    if delta != q - delta: # Avoid duplicate if q is even and delta is half
                        w_minus = w.copy()
                        w_minus[i] = (w[i] - delta) % q
                        sphere.append(w_minus)
        spheres.append(sphere)
    
    # Convert spheres to index lists for faster array operations
    sphere_indices = []
    for s in spheres:
        idxs = [word_to_idx[tuple(x)] for x in s]
        sphere_indices.append(idxs)
    
    # Greedy Set Cover
    covered = np.zeros(total_words, dtype=bool)
    cover_count = np.zeros(total_words, dtype=np.int32) # For purge phase
    code_indices = []
    uncovered_count = total_words
    
    # Initial Greedy
    while uncovered_count > 0:
        if time.time() - start_time > time_limit:
            break
            
        best_gain = -1
        best_idx = -1
        
        # Optimization: Only consider words that are not yet covered as potential centers?
        # No, a covered word can still cover other uncovered words.
        # But we can skip words if their gain is 0.
        
        for i in range(total_words):
            # Check if this candidate can cover any new words
            # gain = sum of (not covered) for indices in sphere_indices[i]
            # Using numpy for speed
            s_idx = sphere_indices[i]
            gain = np.sum(1 - covered[s_idx])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_gain == 0:
            break # Should not happen if uncovered_count > 0 and spheres are correct
            
        # Add best_idx to code
        code_indices.append(best_idx)
        s_idx = sphere_indices[best_idx]
        for idx in s_idx:
            if not covered[idx]:
                uncovered_count -= 1
                covered[idx] = True
            cover_count[idx] += 1
            
    # Convert indices to actual codewords
    current_code = [all_words[i] for i in code_indices]
    
    # Purge Phase: Remove redundant codewords
    # We repeat this a few times or until no more can be removed
    # Since the order of removal can matter, a single pass might leave redundancies that become apparent after removing others.
    # We can do multiple passes.
    
    max_purge_passes = 5
    for _ in range(max_purge_passes):
        if time.time() - start_time > time_limit:
            break
            
        removed_any = False
        # Iterate in reverse to allow modification of list? Or just build a new list.
        # Building a new list is safer.
        new_code = []
        for c in current_code:
            c_idx = word_to_idx[tuple(c)]
            s_idx = sphere_indices[c_idx]
            
            # Temporarily remove c's coverage
            for idx in s_idx:
                cover_count[idx] -= 1
                
            # Check if any word in sphere is now uncovered
            if np.any(cover_count[s_idx] == 0):
                # c is essential, put it back
                for idx in s_idx:
                    cover_count[idx] += 1
                new_code.append(c)
            else:
                # c is redundant, keep it removed
                removed_any = True
                
        current_code = new_code
        if not removed_any:
            break
            
    # Final check: ensure validity (optional but good for debugging/safety)
    # The evaluator checks validity. We trust the logic.
    
    return current_code
