def construct():
    import numpy as np
    import bisect
    
    MAX_N = 10_000_000
    
    # 1. Compute sigma(n) for all n <= MAX_N using NumPy sieve
    # sigma[n] = sum of divisors of n
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes style for sum of divisors
    # for i in 1..N, add i to all multiples of i
    # This is O(N log N)
    for i in range(1, MAX_N + 1):
        sigma[i::i] += i
        
    # 2. Sort indices by sigma value
    # We want to group numbers with the same sigma
    order = np.argsort(sigma)
    sorted_sigmas = sigma[order]
    
    # 3. Iterate through sorted indices to find groups
    best_error_num = 0
    best_error_den = 1
    best_a = -1
    best_b = -1
    
    i = 0
    N = MAX_N
    while i < N + 1:
        current_sigma = sorted_sigmas[i]
        j = i + 1
        
        # Find the end of the block with the same sigma value
        while j < N + 1 and sorted_sigmas[j] == current_sigma:
            j += 1
            
        # The block is order[i:j]
        block_size = j - i
        if block_size >= 2:
            # Extract the numbers in this block
            # These are the indices (values of n) that have this sigma
            block_vals = order[i:j]
            
            # Sort the values in the block to allow binary search
            # block_vals is a NumPy array, we can sort it
            block_vals.sort()
            
            # Convert to Python list for bisect? 
            # bisect works on lists. NumPy arrays don't support bisect directly in the same way?
            # Actually, bisect works on any sequence supporting __getitem__ and __len__.
            # But converting to list might be faster for small blocks?
            # Let's try to use bisect on the NumPy array directly if possible, or convert.
            # bisect.bisect_left expects a list. Let's convert.
            # block_list = block_vals.tolist()
            
            # For each b in the block, find a in the block closest to 1.5 * b
            # We iterate over b
            for b in block_vals:
                target = 1.5 * b
                
                # Binary search in block_vals
                # bisect requires a list. 
                # If block is small, tolist is fine.
                # If block is large, tolist is expensive.
                # Average block size is 1. Max is small.
                # So tolist is fine.
                if block_size < 100:
                    blist = block_vals.tolist()
                    pos = bisect.bisect_left(blist, target)
                    
                    candidates = []
                    if pos < block_size:
                        candidates.append(blist[pos])
                    if pos > 0:
                        candidates.append(blist[pos-1])
                        
                    for a in candidates:
                        if a == b:
                            continue
                        
                        # Error is |2a - 3b| / (2b)
                        num = abs(2 * a - 3 * b)
                        if num == 0:
                            # Invalid, skip
                            continue
                            
                        den = 2 * b
                        
                        # Compare num/den with best_error_num/best_error_den
                        # num * best_error_den < best_error_num * den
                        if best_a == -1 or num * best_error_den < best_error_num * den:
                            best_a = a
                            best_b = b
                            best_error_num = num
                            best_error_den = den
                            
        i = j
        
    if best_a == -1:
        # Fallback, should not happen
        return [11, 6]
        
    return [best_a, best_b]
