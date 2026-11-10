import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Sieve-based computation of small-prime factors
    # small_factor[x] = product of all prime factors of x that are < 24 (with multiplicity)
    # Start with small_factor[x] = x, then for each small prime p, divide out p from all multiples
    
    small_factor = list(range(max_val + 1))
    
    for p in small_primes:
        for multiple in range(p, max_val + 1, p):
            while small_factor[multiple] % p == 0:
                small_factor[multiple] //= p
    
    # Now small_factor[x] is the part of x composed of primes >= 29
    # Wait, that's the opposite! I want the part composed of primes < 24.
    # Let me reconsider.
    
    # Actually, I want the product of prime factors < 24. 
    # If I start with x and divide out all prime factors >= 29, I get the < 24 part.
    # But I don't know the prime factors >= 29 easily.
    
    # Alternative: start with result = 1, and for each small prime p, multiply by p^v_p(x)
    # That's what the original does.
    
    # Let me use a different sieve approach:
    # small_factor[x] starts at 1
    # For each small prime p, for each multiple m of p, multiply small_factor[m] by p^v_p(m)
    
    small_factor = [1] * (max_val + 1)
    
    for p in small_primes:
        for multiple in range(p, max_val + 1, p):
            temp = multiple
            while temp % p == 0:
                small_factor[multiple] *= p
                temp //= p
    
    # Now sliding window scan
    freq = {}
    active_count = 0
    
    # Initialize for n=1000: window is [1001, 1024]
    for i in range(1, 25):
        val = small_factor[1000 + i]
        if val not in freq:
            freq[val] = 0
            active_count += 1
        freq[val] += 1
        
    best_score = active_count
    best_n = 1000
    
    # Slide
    for n in range(1001, max_n + 1):
        if time.time() > deadline:
            break
            
        old_val = small_factor[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            active_count -= 1
            del freq[old_val]
            
        new_val = small_factor[n + 24]
        if new_val not in freq:
            freq[new_val] = 0
            active_count += 1
        freq[new_val] += 1
        
        if active_count < best_score:
            best_score = active_count
            best_n = n
            if best_score == 1:
                break
                
    return best_n
