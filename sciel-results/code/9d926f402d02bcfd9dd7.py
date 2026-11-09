import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # We need to find n in [10000, 1000000] that minimizes the largest prime factor of n*(n+1)
    # Strategy: Precompute smallest prime factor (SPF) up to 1000001 using a sieve.
    # Then for each n, we can quickly compute the largest prime factor of n and n+1.
    # The largest prime factor of n*(n+1) is max(lpf(n), lpf(n+1)).
    
    # Step 1: Sieve for smallest prime factor up to 1000001
    N = 1000001
    spf = list(range(N))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Step 2: Precompute largest prime factor for each number up to N
    # lpf[i] = largest prime factor of i
    lpf = [1] * N
    for i in range(2, N):
        # Factorize i using spf
        x = i
        max_p = 1
        while x > 1:
            p = spf[x]
            max_p = p  # Since we divide by smallest first, the last prime we extract is the largest? 
            # No, that's not right. We need to track the max.
            # Actually, spf gives the smallest prime factor. 
            # To get the largest, we can just factorize completely.
            # But we can optimize: the largest prime factor of i is the last prime in the factorization sequence if we always divide by the smallest.
            # Wait, if we divide by smallest repeatedly, the remaining number is the product of larger primes.
            # Let's just compute it properly for each i, but that's O(N log N) which is fine for N=1e6.
            
            # Better approach: 
            # lpf[i] can be computed recursively or by iterating.
            # Let's use the property: if spf[i] == i, then i is prime, so lpf[i] = i.
            # Otherwise, lpf[i] = max(spf[i], lpf[i // spf[i]])? No.
            # Example: i = 12. spf[12]=2. 12//2 = 6. lpf[6] = 3. max(2,3)=3. Correct.
            # Example: i = 18. spf[18]=2. 18//2=9. lpf[9]=3. max(2,3)=3. Correct.
            # Example: i = 15. spf[15]=3. 15//3=5. lpf[5]=5. max(3,5)=5. Correct.
            # So lpf[i] = max(spf[i], lpf[i // spf[i]]) works.
            pass
        
        # Let's fill lpf array in a single pass
        if spf[i] == i:
            lpf[i] = i
        else:
            lpf[i] = max(spf[i], lpf[i // spf[i]])
    
    # Step 3: Scan all n in [10000, 1000000]
    best_n = 10000
    best_val = float('inf')
    
    for n in range(10000, 1000001):
        if time.time() > deadline:
            break
        # Largest prime factor of n*(n+1) is max(lpf[n], lpf[n+1])
        val = max(lpf[n], lpf[n+1])
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
