import time
import math
from collections import deque

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    # Sieve of Eratosthenes to find smallest prime factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute Largest Prime Factor (LPF) for each number
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        x = i
        max_p = 1
        while x > 1:
            p = spf[x]
            if p > max_p:
                max_p = p
            while x % p == 0:
                x //= p
        lpf[i] = max_p
    
    # Find the pair (a, b) minimizing max(lpf[a], lpf[b])
    best_val = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        if time.time() > deadline:
            break
        b = N - a
        val = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
            # The minimum possible max is at least 3 (since N is odd, one of a,b is even,
            # the other odd; the odd one has an odd prime factor >= 3)
            if best_val == 3:
                break
    
    return [best_a, best_b]
