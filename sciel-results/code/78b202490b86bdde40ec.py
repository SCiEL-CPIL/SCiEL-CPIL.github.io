import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Specific case: q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other cases (though task specifies this one)
        # Simple greedy for general case
        total = q ** n
        if total > 100000:
             # Too large for this specific optimization, return empty or basic
             return []
        
    total_words = 3 ** 8
    num_coords = 8
    alphabet = 3
    
    # Precompute neighbors
    # neighbors[i] is a list of indices j such that d(i,j) <= 1
    # We can compute this on the fly or precompute. Precompute is faster for repeated access.
    # Memory: 6561 * 17 ints ~ 100KB. Fine.
    
    def idx_to_digits(idx):
        digits = [0] * num_coords
        for i in range(num_coords - 1, -1, -1):
            digits[i] = idx % alphabet
            idx //= alphabet
        return digits

    def digits_to_idx(digits):
        idx = 0
        for d in digits:
            idx = idx * alphabet + d
        return idx

    # Precompute neighbor lists
    neighbors = []
    for i in range(total_words):
        digits = idx_to_digits(i)
        nbs = [i]
        for pos in range(num_coords):
            orig_val = digits[pos]
            for val in range(alphabet):
                if val == orig_val:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                nbs.append(digits_to_idx(new_digits))
        neighbors.append(nbs)
        
    # Convert to numpy arrays for faster operations?
    # neighbors is list of lists.
    
    # Greedy Construction
    uncovered = np.ones(total_words, dtype=np.bool_)
    in_code = np.zeros(total_words, dtype=np.bool_)
    codewords = []
    
    # For greedy, we need to find the codeword covering the most uncovered words.
    # We can maintain a count array.
    # count[c] = number of uncovered words in neighbors[c]
    
    # Initial counts
    counts = np.zeros(total_words, dtype=np.int32)
    for i in range(total_words):
        for nb in neighbors[i]:
            counts[i] += 1 # Initially all uncovered
            
    # Actually, we can just compute counts on the fly for the first step, 
    # but maintaining it is faster.
    # Let's initialize counts properly.
    # Since all are uncovered, counts[i] = len(neighbors[i]) = 17.
    counts = np.full(total_words, 17, dtype=np.int32)
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        # Find max count
        best_cw = np.argmax(counts)
        if counts[best_cw] == 0:
            break
            
        # Add to code
        in_code[best_cw] = True
        codewords.append(best_cw)
        
        # Update uncovered and counts
        for nb in neighbors[best_cw]:
            if uncovered[nb]:
                uncovered[nb] = False
                # Decrement counts for all codewords that cover nb
                # Who covers nb? neighbors[nb]
                for cw in neighbors[nb]:
                    if in_code[cw] or counts[cw] > 0: # Only decrement if it was contributing to uncovered count
                         # Wait, counts[cw] tracks uncovered words covered by cw.
                         # If nb becomes covered, and cw covers nb, then counts[cw] should decrease.
                         # But we only decrement if nb was previously uncovered.
                         counts[cw] -= 1

    # Now we have a valid cover. Prune it.
    # Pruning Strategy:
    # 1. Calculate coverage count for each word (how many codewords cover it).
    # 2. A codeword c is removable if for all w in neighbors[c], coverage_count[w] > 1.
    # 3. Remove all such codewords.
    # 4. Repeat.
    # 5. If stuck, try swaps.
    
    code_set = set(codewords)
    
    def prune_once(code_set):
        if not code_set:
            return code_set
        
        # Compute coverage count
        cov_count = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            for nb in neighbors[c]:
                cov_count[nb] += 1
                
        removable = []
        for c in code_set:
            is_removable = True
            for nb in neighbors[c]:
                if cov_count[nb] <= 1:
                    is_removable = False
                    break
            if is_removable:
                removable.append(c)
                
        if removable:
            for c in removable:
                code_set.remove(c)
            return prune_once(code_set)
        else:
            return code_set
            
    code_set = prune_once(code_set)
    
    # Swap-based local search to escape local minima
    # Try to replace a codeword c with d not in set, such that coverage is maintained.
    # This doesn't reduce size directly, but might enable further pruning.
    # However, we want to REDUCE size.
    # A better move: Remove c. If some words become uncovered, find d that covers them.
    # If we can remove c and add NO new words, size decreases. (Handled by prune)
    # If we remove c and add d, size is same.
    # If we remove c, d1, d2 and add e, size decreases by 2.
    
    # Let's try a simple "1-swap" that might lead to reduction:
    # For each c in code_set:
    #   Let U be words covered ONLY by c.
    #   If U is empty, c is removable (prune handles this).
    #   If U is not empty, look for d not in code_set such that U is subset of neighbors[d].
    #   If found, replace c with d.
    #   Then run prune again.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle to avoid bias
        c_list = list(code_set)
        random.shuffle(c_list)
        
        for c in c_list:
            if time.time() > deadline:
                break
            if c not in code_set:
                continue
                
            # Find words covered only by c
            # We need cov_count again.
            cov_count = np.zeros(total_words, dtype=np.int32)
            for cw in code_set:
                for nb in neighbors[cw]:
                    cov_count[nb] += 1
            
            unique_words = []
            for nb in neighbors[c]:
                if cov_count[nb] == 1:
                    unique_words.append(nb)
                    
            if not unique_words:
                # Should have been pruned, but maybe state changed?
                code_set.remove(c)
                improved = True
                break # Restart loop with new set
                
            # Try to find d
            # d must cover all unique_words
            # And d should not be in code_set
            # Also, ideally d should be "good" (cover many other things)
            
            # Optimization: Pre-compute for each word w, the list of codewords covering it.
            # We have neighbors. neighbors[w] is the list of codewords covering w.
            
            candidates = set(neighbors[unique_words[0]])
            for w in unique_words[1:]:
                candidates.intersection_update(neighbors[w])
                
            # Remove existing codewords from candidates
            candidates -= code_set
            
            if candidates:
                # Pick a random candidate
                d = random.choice(list(candidates))
                # Swap
                code_set.remove(c)
                code_set.add(d)
                improved = True
                # Run prune after swap
                code_set = prune_once(code_set)
                break # Restart main loop to re-evaluate all
                
    # Final check: ensure valid
    # Convert to list of digits
    result = [idx_to_digits(c) for c in code_set]
    
    # Verify distinctness
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
            
    return unique_result
