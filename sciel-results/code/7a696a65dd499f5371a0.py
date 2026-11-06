import time

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000000 + 24
    
    # Precompute small prime parts using sieve-style approach
    # small_parts[m] = product of p^v_p(m) for p in primes
    small_parts = [1] * (max_val + 1)
    
    for p in primes:
        p_power = p
        while p_power <= max_val:
            for m in range(p_power, max_val + 1, p_power):
                # Check if m is divisible by p_power but not by p_power * p
                if m % (p_power * p) != 0:
                    small_parts[m] *= p
            p_power *= p
    
    best_n = 1000
    best_score = 25 # Max possible distinct values is 24
    
    # Sliding window approach
    # Window is [n+1, n+24]
    # We start with n=1000, window is [1001, 1024]
    
    # Use an array for counts instead of dict for speed
    # The maximum value of small_parts is max_val
    counts = [0] * (max_val + 1)
    
    distinct_count = 0
    for i in range(1, 25):
        val = small_parts[1000 + i]
        if counts[val] == 0:
            distinct_count += 1
        counts[val] += 1
            
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
            
        # Add element at n+24
        val_in = small_parts[n + 24]
        if counts[val_in] == 0:
            distinct_count += 1
        counts[val_in] += 1
        
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
            if best_score == 1:
                break
                
    return best_n
