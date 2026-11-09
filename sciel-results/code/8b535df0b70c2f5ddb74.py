import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second safety margin
    
    N = 1000000
    WINDOW = 24
    PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23], dtype=np.int64)
    MAX_VAL = N + WINDOW
    
    # Precompute sp[m] = complete factor of m composed of primes < 24
    # Using NumPy vectorized operations
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # For each prime, multiply into all multiples
    for p in PRIMES:
        p = int(p)
        multiples = np.arange(p, MAX_VAL + 1, p, dtype=np.int64)
        # For each multiple, we need to divide out all factors of p
        # This is tricky to vectorize perfectly, so we use a different approach
        # sp[m] should be the product of all prime factors < 24 with multiplicity
        
    # Better approach: compute sp using the smallest prime factor method
    # But vectorized: for each number, we want the product of all small prime factors
    # Let's use a sieve-like approach
    
    # Initialize sp to 1
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # For each prime p, for all multiples m of p, divide m by p repeatedly
    # and multiply sp[m] by p each time
    for p in PRIMES:
        p = int(p)
        # For multiples of p
        m = p
        while m <= MAX_VAL:
            # Divide m by p as many times as possible
            temp = m
            while temp % p == 0:
                sp[m] *= p
                temp //= p
            m += p
    
    # Now sp[m] contains the complete factor of m composed of primes < 24
    # Convert to Python list for easier manipulation, or keep as numpy array
    # Let's keep as numpy array and use it directly
    
    # Sliding window: for each n in [1000, 1000000], window is sp[n+1] to sp[n+24]
    # We need to count distinct values in each window
    
    # To do this efficiently, we can use a frequency dictionary approach
    # But let's try to vectorize where possible
    
    # First, extract the array of sp values we need: sp[1001] to sp[1000024]
    sp_array = sp[1001:1000025]  # indices 0 to 999023, corresponding to numbers 1001 to 1000024
    
    # For window starting at index i (0-indexed in sp_array), we look at sp_array[i:i+24]
    # This corresponds to n = 1000 + i
    # We need n in [1000, 1000000], so i in [0, 999000]
    
    # Use a sliding window with frequency counting
    from collections import defaultdict
    
    best_n = 1000
    best_distinct = WINDOW + 1
    
    freq = defaultdict(int)
    current_distinct = 0
    
    # Initialize window for n=1000 (i=0): sp_array[0:24]
    for i in range(WINDOW):
        val = int(sp_array[i])
        if freq[val] == 0:
            current_distinct += 1
        freq[val] += 1
    
    if current_distinct < best_distinct:
        best_distinct = current_distinct
        best_n = 1000
    
    # Slide the window
    # For i from 1 to 999000 (n = 1000 + i)
    # Remove sp_array[i-1], add sp_array[i+23]
    for i in range(1, 999001):
        if time.time() > deadline:
            break
        
        # Remove sp_array[i-1]
        old_val = int(sp_array[i - 1])
        freq[old_val] -= 1
        if freq[old_val] == 0:
            current_distinct -= 1
            del freq[old_val]
        
        # Add sp_array[i + WINDOW - 1]
        new_val = int(sp_array[i + WINDOW - 1])
        if freq[new_val] == 0:
            current_distinct += 1
        freq[new_val] += 1
        
        if current_distinct < best_distinct:
            best_distinct = current_distinct
            best_n = 1000 + i
            if best_distinct == 1:
                break
    
    return best_n
