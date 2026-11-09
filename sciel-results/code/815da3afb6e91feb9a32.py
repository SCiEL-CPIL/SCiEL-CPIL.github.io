import sys
import math
import time
from collections import defaultdict
import bisect

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin for 90s limit
    
    MAX_N = 10_000_000
    
    # Build SPF (smallest prime factor) sieve
    spf = list(range(MAX_N + 1))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute sigma using multiplicative property with SPF
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    
    for n in range(2, MAX_N + 1):
        if time.time() > deadline - 10:
            break
        p = spf[n]
        m = n // p
        if m % p == 0:
            # n = p^k * r where p does not divide r
            # Find highest power of p dividing n
            pk = 1
            temp = n
            while temp % p == 0:
                pk *= p
                temp //= p
            # Sum of geometric series: 1 + p + p^2 + ... + p^k = (p^(k+1) - 1)/(p - 1)
            # pk = p^k, so p^(k+1) = pk * p
            if p == 1:
                sum_powers = pk
            else:
                sum_powers = (pk * p - 1) // (p - 1)
            sigma[n] = sum_powers * sigma[temp]
        else:
            # n = p * m where p does not divide m
            sigma[n] = (1 + p) * sigma[m]
    
    if time.time() > deadline - 3:
        return [11, 6]
    
    best_err_num = float('inf')
    best_err_den = 1
    best_a = 11
    best_b = 6
    
    # For each b, check a candidates near 1.5*b
    for b in range(1, MAX_N + 1):
        if time.time() > deadline - 2:
            break
        
        target_a = (3 * b) // 2
        sb = sigma[b]
        
        # Check a in a window around target_a
        # |2a - 3b| should be small
        for a in range(max(1, target_a - 20), min(MAX_N, target_a + 20) + 1):
            if a == b:
                continue
            num = abs(2 * a - 3 * b)
            if num == 0:
                continue  # Invalid: error is exactly 0
            den = 2 * b
            
            # Compare num/den with best_err_num/best_err_den using cross multiplication
            if num * best_err_den < best_err_num * den:
                if sigma[a] == sb:
                    best_err_num = num
                    best_err_den = den
                    best_a = a
                    best_b = b
    
    if best_err_num == float('inf'):
        return [11, 6]
    
    return [int(best_a), int(best_b)]
