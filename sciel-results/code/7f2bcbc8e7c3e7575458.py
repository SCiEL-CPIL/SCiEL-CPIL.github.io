import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Precompute small_factor[x] = product of prime factors < 24 (with multiplicity)
    small_factor = [1] * (max_val + 1)
    
    for p in small_primes:
        for multiple in range(p, max_val + 1, p):
            val = multiple
            while val % p == 0:
                small_factor[multiple] *= p
                val //= p
                
    # Sliding window approach
    # Window is [n+1, ..., n+24]
    # We start with n=1000, window [1001, ..., 1024]
    
    best_score = 25
    best_n = 1000
    
    freq = {}
    distinct = 0
    
    # Initialize window for n=1000
    for i in range(1, 25):
        f = small_factor[1000 + i]
        if f not in freq:
            freq[f] = 1
            distinct += 1
        else:
            freq[f] += 1
            
    if distinct < best_score:
        best_score = distinct
        best_n = 1000
        
    # Slide window for n from 1001 to 1000000
    # Remove n, add n+24
    for n in range(1001, 1000001):
        # Remove element at position n (which is the first element of the previous window [n, ..., n+23])
        # Wait, the window for n-1 was [(n-1)+1, ..., (n-1)+24] = [n, ..., n+23]
        # The window for n is [n+1, ..., n+24]
        # So we remove S(n) and add S(n+24)
        
        remove_f = small_factor[n]
        freq[remove_f] -= 1
        if freq[remove_f] == 0:
            del freq[remove_f]
            distinct -= 1
        
        add_f = small_factor[n + 24]
        if add_f not in freq:
            freq[add_f] = 1
            distinct += 1
        else:
            freq[add_f] += 1
            
        if distinct < best_score:
            best_score = distinct
            best_n = n
            if best_score == 1:
                break
                
        # Periodic time check
        if n % 100000 == 0:
            if time.time() > deadline:
                break
                
    return best_n
