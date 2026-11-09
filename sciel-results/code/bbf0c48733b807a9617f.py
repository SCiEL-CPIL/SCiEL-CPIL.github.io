import time
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need smooth parts for numbers up to 1000000 + 24 = 1000024
    max_val = 1000024
    sp = [1] * (max_val + 1)
    
    # Sieve: for each small prime p, multiply sp[m] by p^exp for each multiple m of p
    for p in small_primes:
        for m in range(p, max_val + 1, p):
            temp = m
            exp = 0
            while temp % p == 0:
                exp += 1
                temp //= p
            if exp > 0:
                sp[m] *= (p ** exp)
    
    best_n = 1000
    best_count = float('inf')
    
    # Sliding window over sp[n+1 : n+25] for n in [1000, 1000000]
    # Window for n: indices n+1 to n+24 (inclusive)
    
    # Initialize counter for n=1000: window [1001, 1024]
    cnt = Counter()
    for i in range(1001, 1025):
        cnt[sp[i]] += 1
    
    if len(cnt) < best_count:
        best_count = len(cnt)
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
        cnt[old_val] -= 1
        if cnt[old_val] == 0:
            del cnt[old_val]
        
        # Add new rightmost
        new_val = sp[new_n + 24]
        cnt[new_val] += 1
        
        count = len(cnt)
        if count < best_count:
            best_count = count
            best_n = new_n
            if best_count == 1:
                break
    
    return best_n
