import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    # We need to evaluate n in [1000, 1000000]
    # For each n, we look at n+1, n+2, ..., n+24
    # So we need numbers from 1001 to 1000024
    
    # Precompute bitmask for each number in [1001, 1000024]
    # bitmask[i] represents which of the 9 primes divide (1001 + i)
    # Index 0 corresponds to number 1001, index 1 to 1002, etc.
    
    # Total numbers needed: 1000024 - 1001 + 1 = 999024
    total_nums = 1000024 - 1001 + 1
    
    # Create bitmask array
    bitmasks = [0] * total_nums
    
    # For each prime, mark all multiples in the range
    for pi, p in enumerate(primes):
        bit = 1 << pi
        # Find first multiple of p >= 1001
        first_mult = ((1001 + p - 1) // p) * p
        for num in range(first_mult, 1000025, p):
            idx = num - 1001
            if idx < total_nums:
                bitmasks[idx] |= bit
    
    # Now use sliding window of size 24
    # Window starting at position j (0-indexed) covers bitmasks[j:j+24]
    # This corresponds to numbers (1001+j) to (1001+j+23)
    # Which is n+1 to n+24 where n = 1000 + j
    # So n = 1000 + j, and we need n in [1000, 1000000]
    # j ranges from 0 to total_nums - 24 = 999024 - 24 = 998999
    # n = 1000 + j ranges from 1000 to 1000 + 998999 = 999999
    # Wait, that's only up to 999999. Let me recheck.
    # n+24 <= 1000024 means n <= 1000000. 
    # n+1 >= 1001 means n >= 1000.
    # So n in [1000, 1000000], j = n - 1000, j in [0, 1000000-1000] = [0, 999000]
    # Window j covers bitmasks[j] to bitmasks[j+23], which are numbers 1001+j to 1001+j+23
    # For j=999000: numbers 1001+999000=1000001 to 1000001+23=1000024. Correct.
    
    # Sliding window with frequency count
    # freq[pi] = count of numbers in current window divisible by prime pi
    # distinct_count = number of pi where freq[pi] > 0
    
    freq = [0] * num_primes
    distinct_count = 0
    
    # Initialize first window (j=0, n=1000)
    for k in range(24):
        bm = bitmasks[k]
        for pi in range(num_primes):
            if bm & (1 << pi):
                freq[pi] += 1
                if freq[pi] == 1:
                    distinct_count += 1
    
    best_n = 1000
    best_score = distinct_count
    
    # Slide the window
    # j goes from 1 to 999000 (n goes from 1001 to 1000000)
    max_j = 999000  # j = n - 1000, n max = 1000000
    
    for j in range(1, max_j + 1):
        if time.time() > deadline:
            break
        
        # Remove bitmasks[j-1] from window
        bm_out = bitmasks[j - 1]
        for pi in range(num_primes):
            if bm_out & (1 << pi):
                freq[pi] -= 1
                if freq[pi] == 0:
                    distinct_count -= 1
        
        # Add bitmasks[j+23] to window
        bm_in = bitmasks[j + 23]
        for pi in range(num_primes):
            if bm_in & (1 << pi):
                freq[pi] += 1
                if freq[pi] == 1:
                    distinct_count += 1
        
        n = 1000 + j
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
    
    return best_n
