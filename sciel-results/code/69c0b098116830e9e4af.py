"""Smooth two-summand decomposition of a prescribed integer."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    N = 123457
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Compute smallest prime factor (SPF) for all numbers up to N
    spf = list(range(N + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute largest prime factor (LPF) for all numbers up to N
    lpf = [0] * (N + 1)
    lpf[0] = 0
    lpf[1] = 1
    for i in range(2, N + 1):
        n = i
        max_pf = 1
        while n > 1:
            p = spf[n]
            max_pf = p
            while n % p == 0:
                n //= p
        lpf[i] = max_pf
    
    # Now search for the pair (a, b) with a+b=N minimizing max(lpf[a], lpf[b])
    # Strategy: sort numbers by their LPF value, then try to find a pair
    # whose LPFs are both small.
    
    # Approach: binary search on the threshold T. Find the smallest T such that
    # there exists a pair (a, b) with a+b=N, lpf[a] <= T, lpf[b] <= T.
    
    # First, let's collect all numbers and their LPF values
    # We want to find min over a in [1, N-1] of max(lpf[a], lpf[N-a])
    
    # Direct O(N) scan:
    best_val = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
        if time.time() > deadline:
            break
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
