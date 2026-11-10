"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    # Primes strictly below 24: 2, 3, 5, 7, 11, 13, 17, 19, 23
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    # Precompute small prime powers for fast factor extraction
    # For each prime, precompute powers up to 1000000+24
    max_val = 1000000 + 24
    prime_powers = []
    for p in primes:
        powers = []
        pk = 1
        while pk <= max_val:
            powers.append(pk)
            pk *= p
        prime_powers.append(powers)
    
    def small_prime_factor_mask(x):
        """Return a bitmask of which small primes divide x, and the extracted factor."""
        mask = 0
        factor = 1
        for i, p in enumerate(primes):
            if x % p == 0:
                mask |= (1 << i)
                # Extract complete power of p from x
                temp = x
                while temp % p == 0:
                    factor *= p
                    temp //= p
        return mask, factor
    
    def count_distinct_factors_in_window(n):
        """Count distinct small-prime factor masks in n+1 to n+24."""
        masks = set()
        for i in range(1, 25):
            x = n + i
            mask, _ = small_prime_factor_mask(x)
            masks.add(mask)
        return len(masks)
    
    # Strategy: sliding window approach
    # For each position n in [1000, 1000000-24], compute the number of distinct masks
    # in n+1 to n+24, and find the minimum.
    
    # Precompute masks for all relevant numbers
    # We need masks for numbers from 1001 to 1000000+24
    min_start = 1001
    max_end = 1000000 + 24
    
    # Compute mask for each number in range
    # Use a dictionary or array
    num_range = max_end - min_start + 1
    masks_arr = [0] * (num_range + 1)
    
    for x in range(min_start, max_end + 1):
        mask = 0
        for i, p in enumerate(primes):
            if x % p == 0:
                mask |= (1 << i)
        masks_arr[x - min_start] = mask
    
    # Sliding window: for each n, window is n+1 to n+24
    # In masks_arr, this corresponds to indices (n+1-min_start) to (n+24-min_start)
    # n ranges from 1000 to 999976
    
    best_n = 1000
    best_count = float('inf')
    
    # Use a frequency array for masks in the current window
    # Mask values range from 0 to 2^9-1 = 511
    freq = [0] * 512
    
    # Initialize window for n=1000: numbers 1001 to 1024
    for i in range(1, 25):
        idx = (1000 + i) - min_start
        masks_arr_local = masks_arr[idx]
        freq[masks_arr_local] += 1
    
    # Count distinct masks in initial window
    distinct = sum(1 for f in freq if f > 0)
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide the window
    # For n from 1001 to 999976:
    # Remove n (which is n-1+1 = n) from window, add n+24
    for n in range(1001, 999977):
        # Check deadline
        if time.time() > deadline - 2:
            break
        
        # Remove element at position n (which was n-1+1 = n)
        # In the window for n-1, we had (n-1)+1 to (n-1)+24 = n to n+23
        # For window n, we need n+1 to n+24
        # So remove n, add n+24
        
        # Remove n
        idx_remove = n - min_start
        mask_remove = masks_arr[idx_remove]
        freq[mask_remove] -= 1
        if freq[mask_remove] == 0:
            distinct -= 1
        
        # Add n+24
        idx_add = (n + 24) - min_start
        mask_add = masks_arr[idx_add]
        if freq[mask_add] == 0:
            distinct += 1
        freq[mask_add] += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
