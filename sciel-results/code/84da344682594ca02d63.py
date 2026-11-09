import math

def construct():
    # Target: find the longest run of consecutive integers (up to 200) with distinct totients,
    # within [1, 10^12].
    # Strategy:
    # 1. Precompute primes up to sqrt(10^12) = 10^6.
    # 2. Use a segmented sieve to compute phi(n) for a large contiguous block of numbers.
    #    The block size is chosen to be large enough to likely contain a run of length 200,
    #    but small enough to fit in memory and compute quickly.
    # 3. Slide a window of size 200 over the computed phi values to check for distinctness.
    #    Use a frequency count to efficiently check if all values in the window are distinct.
    # 4. If a run of 200 is found, return it. Otherwise, return the best run found.

    N_MAX = 10**12
    MAX_K = 200
    
    # 1. Sieve primes up to sqrt(N_MAX)
    limit = int(math.isqrt(N_MAX))
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 2. Segmented phi computation
    # We will scan from a high number to increase the chance of finding long runs,
    # as numbers with many small factors (which cause collisions in phi) become rarer?
    # Actually, phi(n) collisions are related to the structure of n.
    # Let's scan a large block, e.g., 1,000,000 numbers.
    # We can scan multiple blocks if needed, but 120s limit suggests we should be careful.
    # Let's try a block of size 5,000,000.
    # sqrt(10^12) = 10^6.
    # Number of primes up to 10^6 is ~78,000.
    # For each prime p, we update ~ L/p entries.
    # Sum_{p <= 10^6} 1/p ~ log log 10^6 ~ 2.6.
    # So ~ 2.6 * L operations. For L=5*10^6, that's 1.3 * 10^7 operations. Very fast in Python?
    # Python loops are slow. 10^7 operations might take ~1-5 seconds.
    # Let's optimize the inner loop.
    
    block_size = 10**6
    # Start scanning from a high range, e.g., 10^11
    # We need s + k - 1 <= 10^12
    # Let's start at 10^11 and go up.
    # If we don't find 200, we might try other ranges.
    # But usually, a run of 200 exists.
    
    best_k = 1
    best_s = 1
    
    # To maximize performance, we process a large block at once.
    # Let's pick a start point.
    # Random start might be good to avoid edge cases or specific structures.
    # But deterministic is better for reproducibility if we don't know the best spot.
    # Let's try a few specific blocks.
    
    # We will scan from 10^11 to 10^11 + 10^7 in steps of block_size.
    # Or just one big block of 10^7?
    # Memory: 10^7 integers. Python ints are 28 bytes. 280 MB. Might be too much.
    # Let's stick to 10^6 or 2*10^6.
    
    # We'll try scanning a few segments.
    # Segment 1: 10^11 to 10^11 + 10^6
    # Segment 2: 10^12 - 10^6 to 10^12
    
    # Actually, let's just scan from 10^11 upwards for a few blocks.
    # If we find k=200, we stop.
    
    for block_start in [10**11, 10**11 + 10**6, 10**12 - 10**6]:
        if block_start + block_size > N_MAX:
            block_end = N_MAX
        else:
            block_end = block_start + block_size
            
        n_items = block_end - block_start + 1
        if n_items <= 0:
            continue
            
        # Initialize arrays
        phi = list(range(block_start, block_end + 1))
        rem = list(range(block_start, block_end + 1))
        
        # Factorization using primes
        for p in primes:
            if p * p > block_end:
                break
            # Find the first multiple of p in [block_start, block_end]
            first_multiple = ((block_start + p - 1) // p) * p
            for multiple in range(first_multiple, block_end + 1, p):
                idx = multiple - block_start
                if rem[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while rem[idx] % p == 0:
                        rem[idx] //= p
        
        # Handle remaining prime factors
        for i in range(n_items):
            if rem[i] > 1:
                p = rem[i]
                phi[i] -= phi[i] // p
        
        # Sliding window to find max distinct run
        # We want max k <= 200.
        # We can just check windows of size 200. If found, return immediately.
        # If not, we need to find the max k < 200.
        # To find max k, we can use a sliding window with frequency counts.
        
        # Let's first check for k=200.
        found_200 = False
        if n_items >= 200:
            # Count frequencies in first window
            freq = {}
            for i in range(200):
                val = phi[i]
                freq[val] = freq.get(val, 0) + 1
            
            has_dup = any(c > 1 for c in freq.values())
            if not has_dup:
                return {'start': block_start, 'length': 200}
            
            # Slide window
            for i in range(200, n_items):
                # Add new
                val_in = phi[i]
                freq[val_in] = freq.get(val_in, 0) + 1
                # Remove old
                val_out = phi[i - 200]
                freq[val_out] -= 1
                if freq[val_out] == 0:
                    del freq[val_out]
                
                # Check if all counts are 1
                # This is O(window_size) if we check every time.
                # Better to maintain a count of "duplicates" or "unique" count.
                # Let's maintain `num_unique` and `dups` (number of values with count > 1).
                # Actually, simpler: maintain a set of values with count > 1?
                # Or just check if any count > 1.
                # With window size 200, checking 200 items is fast.
                # But we do it 10^6 times. 10^6 * 200 = 2*10^8. Too slow in Python.
                
                # Optimization: Maintain `bad_count` = number of keys with freq > 1.
                # When adding val_in:
                #   if freq[val_in] was 1, it becomes 2 -> bad_count += 1
                #   if freq[val_in] was 2, it becomes 3 -> bad_count unchanged
                # When removing val_out:
                #   if freq[val_out] was 2, it becomes 1 -> bad_count -= 1
                #   if freq[val_out] was 3, it becomes 2 -> bad_count unchanged
                
                # Let's re-implement the sliding window logic with this optimization.
                # We already have the first window processed.
                # Let's restart the sliding window logic for this block properly.
                
                pass # The above loop is incomplete/inefficient. Let's rewrite the window check.
        
        # Proper sliding window implementation
        if n_items >= 2:
            freq = {}
            bad_count = 0
            
            # Initialize first window of size min(200, n_items)
            win_size = min(200, n_items)
            for i in range(win_size):
                val = phi[i]
                if val in freq:
                    if freq[val] == 1:
                        bad_count += 1
                    freq[val] += 1
                else:
                    freq[val] = 1
            
            if bad_count == 0 and win_size == 200:
                return {'start': block_start, 'length': 200}
            
            if bad_count == 0:
                if win_size > best_k:
                    best_k = win_size
                    best_s = block_start
            
            # Slide
            for i in range(win_size, n_items):
                # Add phi[i]
                val_in = phi[i]
                if val_in in freq:
                    if freq[val_in] == 1:
                        bad_count += 1
                    freq[val_in] += 1
                else:
                    freq[val_in] = 1
                
                # Remove phi[i - win_size]
                val_out = phi[i - win_size]
                freq[val_out] -= 1
                if freq[val_out] == 0:
                    del freq[val_out]
                else:
                    if freq[val_out] == 1:
                        bad_count -= 1
                
                if bad_count == 0:
                    # Window [i - win_size + 1, i] is valid
                    start_idx = i - win_size + 1
                    start_val = block_start + start_idx
                    if win_size > best_k:
                        best_k = win_size
                        best_s = start_val
                    if best_k == 200:
                        return {'start': best_s, 'length': 200}
        
    # If no 200 found, return best found
    return {'start': best_s, 'length': best_k}
