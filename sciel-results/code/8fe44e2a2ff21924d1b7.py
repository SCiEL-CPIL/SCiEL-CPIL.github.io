import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    small_prime_set = set(small_primes)
    
    # Precompute the "small prime part" for each number up to 1000024
    # small_part[m] = largest divisor of m composed only of primes < 24
    max_val = 1000024
    small_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        # For each power of p, multiply into small_part where divisible
        power = p
        while power <= max_val:
            # All multiples of power that are divisible by p should get p in their small part
            # But we need to be careful: we want the complete factor of primes < 24
            # Better approach: for each number, extract all small prime factors
            break
    
    # More efficient: for each number, compute its small prime part directly
    # small_part[i] = product of p^a where p < 24 and p^a divides i
    small_part = np.ones(max_val + 1, dtype=np.int64)
    for p in small_primes:
        for multiple in range(p, max_val + 1, p):
            # Count how many times p divides multiple
            temp = multiple
            while temp % p == 0:
                small_part[multiple] *= p
                temp //= p
    
    # Now for each n in [1000, 1000000], we need to count distinct values
    # in small_part[n+1], small_part[n+2], ..., small_part[n+24]
    
    # Use a sliding window approach with a frequency array
    # The small_part values can be large, so we need to track distinct counts
    
    # First, let's find a good candidate by scanning
    # We'll use a hash-based approach for the sliding window
    
    best_n = 1000
    best_count = float('inf')
    
    # Precompute small_part as a list for faster access
    sp = small_part  # numpy array
    
    # For efficiency, let's scan all n and use a set for each window
    # But that's O(24 * range) which is fine for range ~10^6
    
    # Actually, let's be smarter. We can precompute for each position
    # what the small_part is, then slide a window.
    
    # Let's just do a direct scan with a set, but optimize by early termination
    # if we already have too many distinct values
    
    # For a 24-window, the max distinct count is 24.
    # We want to minimize this.
    
    # Let's scan and track the minimum
    window_size = 24
    n_min = 1000
    n_max = 1000000
    
    # Use a counter-based sliding window
    # small_part values: we need to track frequency of each value in the window
    
    # Initialize window for n = n_min
    from collections import defaultdict
    
    freq = defaultdict(int)
    for i in range(1, window_size + 1):
        val = int(sp[n_min + i])
        freq[val] += 1
    
    distinct_count = len(freq)
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = n_min
    
    # Slide the window
    for n in range(n_min + 1, n_max + 1):
        # Remove n-1+1 = n (the element leaving: position n)
        # Window was [n-1+1, n-1+24] = [n, n+23]
        # New window: [n+1, n+24]
        # Leaving: sp[n], Entering: sp[n+24]
        
        leave_val = int(sp[n])
        enter_val = int(sp[n + 24])
        
        freq[leave_val] -= 1
        if freq[leave_val] == 0:
            del freq[leave_val]
        
        freq[enter_val] += 1
        
        distinct_count = len(freq)
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
        
        # Check time
        if time.time() > deadline:
            break
    
    return best_n
