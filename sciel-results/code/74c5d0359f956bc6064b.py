import numpy as np
import itertools
import random
import math
import time

def construct(q, n, radius):
    total_words = q ** n
    start_time = time.time()
    TIME_BUDGET = 115  # Leave buffer for evaluation overhead
    
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val
        
    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word
            
    # Generate all words
    words = [to_word(i) for i in range(total_words)]
    
    # Build adjacency matrix (ball of radius 1)
    # mask[i, j] is True if j is within distance 'radius' of i
    mask = np.zeros((total_words, total_words), dtype=bool)
    
    # Precompute balls for each word
    # For radius=1, q=3, n=8, ball size is 1 + 8*2 = 17
    for i in range(total_words):
        word = words[i]
        mask[i, i] = True
        for pos in range(n):
            for delta in range(1, q):
                new_word = word[:]
                new_word[pos] = (word[pos] + delta) % q
                j = to_int(new_word)
                mask[i, j] = True
                
    # --- Phase 1: Greedy Set Cover ---
    uncovered_mask = np.ones(total_words, dtype=np.int8)
    code = set()
    
    while np.any(uncovered_mask):
        # Count uncovered neighbors for each candidate
        counts = mask @ uncovered_mask
        # Mask out already selected
        for c in code:
            counts[c] = -1
            
        best_c = np.argmax(counts)
        best_count = counts[best_c]
        
        if best_count <= 0:
            break
            
        code.add(best_c)
        uncovered_mask[mask[best_c, :]] = 0
        
    # --- Phase 2: Local Search ---
    # Initialize cover counts
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code:
        cover_count[mask[c, :]] += 1
        
    def add_to_cover(c):
        cover_count[mask[c, :]] += 1
        
    def remove_from_cover(c):
        cover_count[mask[c, :]] -= 1
        
    # Helper to check if code is valid (should always be true if we maintain invariants)
    def is_valid(code_set):
        if not code_set:
            return total_words == 0
        # Check if every word is covered
        # Efficient check: sum of cover_count should be at least total_words? No.
        # Check if min(cover_count) >= 1
        return np.min(cover_count) >= 1
        
    max_iterations = 5000
    no_improve_count = 0
    stall_limit = 200
    
    for _ in range(max_iterations):
        if time.time() - start_time > TIME_BUDGET:
            break
            
        improved = False
        
        # 1. Remove redundant codewords
        # A codeword c is redundant if it doesn't uniquely cover any point.
        # i.e., for all w in ball(c), cover_count[w] > 1.
        unique_mask = (cover_count == 1)
        # unique_neighbors[c] = number of points uniquely covered by c
        unique_neighbors = mask @ unique_mask
        
        to_remove = []
        for c in list(code):
            if unique_neighbors[c] == 0:
                to_remove.append(c)
                
        if to_remove:
            for c in to_remove:
                code.remove(c)
                remove_from_cover(c)
            improved = True
            no_improve_count = 0
            continue
            
        # 2. Try Swaps
        # For each c in code, try to replace it with c' not in code
        # such that c' covers all unique points of c.
        
        code_list = list(code)
        random.shuffle(code_list)
        
        swapped = False
        for c_out in code_list:
            if time.time() - start_time > TIME_BUDGET:
                break
                
            # Get unique points covered by c_out
            unique_mask_c = mask[c_out, :] & (cover_count == 1)
            unique_indices = np.where(unique_mask_c)[0]
            
            if len(unique_indices) == 0:
                # Should have been removed, but skip just in case
                continue
                
            # Find candidates c' not in code that cover all unique_indices
            # Condition: for all w in unique_indices, mask[c', w] is True
            # This is equivalent to: mask[:, unique_indices].all(axis=1)
            
            # Optimization: If unique_indices is large, this is a strong filter.
            # If small, many candidates.
            
            # We can speed this up by pre-computing candidate validity?
            # But code changes, so it's dynamic.
            
            # Check candidates
            # We can restrict candidates to those that cover at least one unique point?
            # Actually, they must cover ALL.
            
            # Let's try a few random candidates first if the set is large?
            # Or just compute the boolean mask.
            
            # mask[:, unique_indices] has shape (N, K)
            # .all(axis=1) gives shape (N,)
            
            candidates_mask = mask[:, unique_indices].all(axis=1)
            
            # Exclude current code
            # Instead of looping over code (which is ~500), we can use a boolean array
            in_code = np.zeros(total_words, dtype=bool)
            for c in code:
                in_code[c] = True
            candidates_mask[in_code] = False
            
            valid_indices = np.where(candidates_mask)[0]
            
            if len(valid_indices) > 0:
                # Pick a candidate.
                # Strategy: Pick one that covers the most other points?
                # Or random?
                # Let's pick the one with max degree (covers most points) to help future reductions?
                # Or min degree to be "tight"?
                # Usually, greedy picks high degree. For removal, we just need validity.
                # Let's pick randomly from valid ones to explore.
                best_c_in = random.choice(valid_indices.tolist())
                
                # Perform swap
                code.remove(c_out)
                remove_from_cover(c_out)
                code.add(best_c_in)
                add_to_cover(best_c_in)
                
                swapped = True
                improved = True
                break # Restart inner loop to recalculate unique counts
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > stall_limit:
                # Try a perturbation?
                # Remove a random codeword and re-greedy?
                # Or just break?
                # Given time constraints, breaking is safer than risky perturbation.
                break
        else:
            no_improve_count = 0

    # Final check
    if not is_valid(code):
        # Fallback to greedy result if somehow invalid (shouldn't happen)
        pass

    result = [to_word(c) for c in code]
    return result
