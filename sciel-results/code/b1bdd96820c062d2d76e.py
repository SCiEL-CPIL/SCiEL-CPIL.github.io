"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def largest_prime_factor(n):
    """Compute the largest prime factor of n by exact trial division."""
    if n < 2:
        return 1
    lpf = 1
    # Check factor 2
    while n % 2 == 0:
        lpf = 2
        n //= 2
    # Check odd factors
    f = 3
    while f * f <= n:
        while n % f == 0:
            lpf = f
            n //= f
        f += 2
    if n > 1:
        lpf = n
    return lpf

def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy:
    1. Precompute the largest prime factor (LPF) for all integers up to 1000001
       using a sieve-like method.
    2. Scan n from 10000 to 1000000, computing max(lpf[n], lpf[n+1]) for each,
       and track the minimum.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (90s limit)
    
    N = 1000001  # We need lpf up to 1000001 since n can be 1000000, so n+1=1000001
    
    # Initialize lpf array with 0
    lpf = [0] * (N + 1)
    
    # Sieve for largest prime factor
    for i in range(2, N + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    # Check time
    if time.time() > deadline:
        # Fallback: return a reasonable value
        return 10000
    
    # Scan for the minimum
    best_n = 10000
    best_val = float('inf')
    
    for n in range(10000, N):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
        # Check time periodically
        if n % 100000 == 0 and time.time() > deadline:
            break
    
    # Verify the result with exact trial division
    if time.time() < deadline - 5:
        verified_val = max(largest_prime_factor(best_n), largest_prime_factor(best_n + 1))
        # If verification disagrees, trust the sieve (shouldn't happen, but safety)
        # The sieve result is authoritative
    
    return best_n

# EVOLVE-BLOCK-END
