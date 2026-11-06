import time

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000000 + 24
    
    # Precompute small prime parts
    # small_parts[m] = product of p^v_p(m) for p in primes
    small_parts = [1] * (max_val + 1)
    
    for p in primes:
        # Iterate through multiples of p
        for m in range(p, max_val + 1, p):
            temp = m
            while temp % p == 0:
                small_parts[m] *= p
                temp //= p
                
    best_n = 1000
    best_score = 25 # Max possible distinct values is 24
    
    # Sliding window approach
    # Window is [n+1, n+24]
    # We start with n=1000, window is [1001, 1024]
    
    # Initialize counts for the first window
    counts = {}
    for i in range(1, 25):
        val = small_parts[1000 + i]
        if val in counts:
            counts[val] += 1
        else:
            counts[val] = 1
            
    distinct_count = len(counts)
    
    if distinct_count < best_score:
        best_score = distinct_count
        best_n = 1000
        
    start_time = time.time()
    deadline = start_time + 50
    
    # Iterate n from 1001 to 1000000
    # Window for n is [n+1, n+24]
    # Previous window for n-1 was [n, n+23]
    # So we remove n, add n+24
    
    for n in range(1001, 1000001):
        if time.time() > deadline:
            break
            
        # Remove element at n (which was n-1 + 1)
        val_out = small_parts[n]
        counts[val_out] -= 1
        if counts[val_out] == 0:
            distinct_count -= 1
            # Optional: del counts[val_out] to save memory, but dict size is small
            # Keeping it at 0 is fine for logic, just check > 0 for distinct count update
            # Actually, if we don't delete, `len(counts)` won't work directly if we rely on keys.
            # We are maintaining `distinct_count` manually, so we don't need `len(counts)`.
            
        # Add element at n+24
        val_in = small_parts[n + 24]
        if counts.get(val_in, 0) == 0:
            distinct_count += 1
        counts[val_in] = counts.get(val_in, 0) + 1
        
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
            if best_score == 1:
                break
                
    return best_n
