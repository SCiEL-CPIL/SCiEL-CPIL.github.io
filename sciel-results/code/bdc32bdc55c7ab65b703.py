import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters, though task specifies this one
        # Simple direct sum or greedy for generic case
        pass 

    # Specific optimization for q=3, n=8, radius=1
    total_words = q ** n
    # Precompute neighbors masks
    # Each word is an integer 0..6560
    # neighbors[w] is a bitmask of words within distance 1 of w
    
    # Helper to convert int to base-3 digits
    def int_to_base3(x, n):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % 3
            x //= 3
        return digits
        
    def base3_to_int(digits):
        x = 0
        for i in range(n-1, -1, -1):
            x = x * 3 + digits[i]
        return x

    neighbors_masks = [0] * total_words
    
    # Precomputation
    for w in range(total_words):
        digits = int_to_base3(w, n)
        mask = 1 << w  # Self
        
        # Change 1 position
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new digits
                new_digits = digits[:]
                new_digits[pos] = new_val
                new_w = base3_to_int(new_digits)
                mask |= (1 << new_w)
        
        neighbors_masks[w] = mask

    def greedy_cover(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = None
            
        uncovered = (1 << total_words) - 1
        code = []
        
        while uncovered:
            best_c = -1
            best_count = -1
            # To handle ties randomly or deterministically
            # Iterating all 6561 candidates
            # Optimization: If we just want any max, we can break early if we find a perfect cover? No, max is 17.
            for c in range(total_words):
                # Count uncovered words covered by c
                count = (uncovered & neighbors_masks[c]).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
                    if best_count == 17: # Max possible coverage
                        # We could break, but there might be other 17s. 
                        # For greedy, any max is fine.
                        # But for diversity in restarts, we might want to pick randomly among maxes.
                        # For now, pick first found.
                        pass
                
            if best_c == -1:
                break # Should not happen if uncovered is not empty
                
            code.append(best_c)
            uncovered &= ~neighbors_masks[best_c]
            
            # Time check
            if time.time() > deadline - 5:
                break
                
        return code

    def local_search_remove(code):
        # code is a list of integers
        if not code:
            return code
            
        # cover_count[w] = number of codewords in code that cover w
        cover_count = [0] * total_words
        
        # Build cover counts
        for c in code:
            # Iterate through bits set in neighbors_masks[c]
            mask = neighbors_masks[c]
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                mask ^= lsb

        changed = True
        # We might want to iterate in a specific order to find redundant codewords
        # A codeword c is redundant if for all w in neighbors[c], cover_count[w] > 1
        
        while changed:
            changed = False
            # Iterate through current code
            # We need to be careful modifying list while iterating
            # Let's collect indices to remove
            to_remove = []
            
            for i, c in enumerate(code):
                is_redundant = True
                mask = neighbors_masks[c]
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] <= 1:
                        is_redundant = False
                        break
                    mask ^= lsb
                
                if is_redundant:
                    to_remove.append(i)
            
            if to_remove:
                # Remove them
                # Decrement cover counts
                for i in to_remove:
                    c = code[i]
                    mask = neighbors_masks[c]
                    while mask:
                        lsb = mask & -mask
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        mask ^= lsb
                
                # Actually remove from list
                # Reverse to keep indices valid if we removed from end? 
                # Or just filter
                to_remove_set = set(to_remove)
                code = [c for i, c in enumerate(code) if i not in to_remove_set]
                changed = True
                
                if time.time() > deadline - 5:
                    break

        return code

    best_code = []
    best_len = float('inf')
    
    # Run a few restarts
    # The greedy is deterministic if no randomness in tie-breaking.
    # Let's add randomness to tie-breaking to explore different solutions.
    
    def greedy_cover_random(seed):
        rng = random.Random(seed)
        uncovered = (1 << total_words) - 1
        code = []
        
        while uncovered:
            best_count = -1
            candidates = []
            for c in range(total_words):
                count = (uncovered & neighbors_masks[c]).bit_count()
                if count > best_count:
                    best_count = count
                    candidates = [c]
                elif count == best_count:
                    candidates.append(c)
            
            if not candidates:
                break
                
            best_c = rng.choice(candidates)
            code.append(best_c)
            uncovered &= ~neighbors_masks[best_c]
            
            if time.time() > deadline - 5:
                break
                
        return code

    # Try a deterministic one first
    code = greedy_cover(None)
    code = local_search_remove(code)
    if len(code) < best_len:
        best_len = len(code)
        best_code = code
        
    # Try a few random restarts
    for i in range(10):
        if time.time() > deadline - 10:
            break
            
        code = greedy_cover_random(i)
        code = local_search_remove(code)
        if len(code) < best_len:
            best_len = len(code)
            best_code = code
            
    # Convert best_code to list of lists
    result = []
    for c in best_code:
        result.append(int_to_base3(c, n))
        
    return result
