import time

def construct():
    start_time = time.time()
    deadline = start_time + 88  # 2s safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need smooth parts for numbers up to 1000000 + 24 = 1000024
    max_val = 1000024
    sp = [1] * (max_val + 1)
    
    # Sieve: for each small prime p, multiply sp[m] by p^exp for each multiple m of p
    # Precompute powers to avoid repeated p ** exp
    for p in small_primes:
        powers = [1]
        while powers[-1] * p <= max_val:
            powers.append(powers[-1] * p)
        for m in range(p, max_val + 1, p):
            temp = m
            exp = 0
            while temp % p == 0:
                exp += 1
                temp //= p
            if exp > 0:
                sp[m] *= powers[exp]
    
    best_n = 1000
    best_count = float('inf')
    
    # Sliding window over sp[n+1 : n+25] for n in [1000, 1000000]
    # Window for n: indices n+1 to n+24 (inclusive)
    
    # Initialize counter for n=1000: window [1001, 1024]
    freq = {}
    distinct_count = 0
    
    for i in range(1001, 1025):
        val = sp[i]
        if val not in freq:
            freq[val] = 1
            distinct_count += 1
        else:
            freq[val] += 1
            
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = 1000
    
    if best_count == 1:
        return best_n
    
    # Slide window: for new_n from 1001 to 1000000
    # Old window (for new_n-1): [new_n, new_n+23]
    # New window (for new_n): [new_n+1, new_n+24]
    # Remove sp[new_n], add sp[new_n+24]
    for new_n in range(1001, 1000001):
        if time.time() > deadline:
            break
        
        # Remove leftmost
        old_val = sp[new_n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            distinct_count -= 1
            
        # Add new rightmost
        new_val = sp[new_n + 24]
        if new_val not in freq:
            freq[new_val] = 1
            distinct_count += 1
        else:
            if freq[new_val] == 0:
                distinct_count += 1
            freq[new_val] += 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = new_n
            if best_count == 1:
                break
    
    return best_n
