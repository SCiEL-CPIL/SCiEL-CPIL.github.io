"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    Uses a sieve-based approach to precompute the largest prime factor (LPF) for all
    numbers up to 1000000, then scans for the optimal pair.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin out of 90s limit
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Sieve for largest prime factor
    # lpf[i] = largest prime factor of i
    lpf = [0] * (MAX_N + 1)
    
    for i in range(2, MAX_N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, MAX_N + 1, i):
                lpf[j] = i
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: return a reasonable default
        return MIN_N
    
    # Now scan all n in [MIN_N, MAX_N-1] (since we need n and n+1, n+1 <= MAX_N)
    # Actually n can go up to MAX_N, and n+1 can be up to MAX_N+1
    # But our sieve only goes up to MAX_N. Let's handle n+1 = MAX_N+1 separately.
    # For n in [MIN_N, MAX_N-1]: both n and n+1 are <= MAX_N
    # For n = MAX_N: n+1 = MAX_N+1 = 1000001. We need LPF of 1000001.
    
    best_n = MIN_N
    best_lpf = lpf[MIN_N] * 0  # Initialize to 0 so any valid pair beats it
    # Actually, let's initialize properly
    best_lpf = float('inf')
    
    for n in range(MIN_N, MAX_N):
        # n and n+1 are both <= MAX_N
        lpf_n = lpf[n]
        lpf_np1 = lpf[n + 1]
        max_lpf = max(lpf_n, lpf_np1)
        if max_lpf < best_lpf:
            best_lpf = max_lpf
            best_n = n
        
        # Check deadline periodically
        if n % 100000 == 0 and time.time() > deadline:
            break
    
    # Handle n = MAX_N = 1000000, where n+1 = 1000001
    if time.time() < deadline:
        n = MAX_N
        lpf_n = lpf[n]  # lpf[1000000]
        # Need LPF of 1000001
        # 1000001 = 101 * 9901? Let's compute via trial division
        # 1000001 / 7 = 142857.28... no
        # 1000001 = 101 * 9901? 101*9901 = 1000001. Yes!
        # 9901 is prime? sqrt(9901) ~ 99.5
        # Check primes up to 99: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97
        # 9901 / 7 = 1414.4... no
        # 9901 / 11 = 900.09... no
        # 9901 / 13 = 761.6... no
        # 9901 / 17 = 582.4... no
        # 9901 / 19 = 521.1... no
        # 9901 / 23 = 430.4... no
        # 9901 / 29 = 341.4... no
        # 9901 / 31 = 319.3... no
        # 9901 / 37 = 267.5... no
        # 9901 / 41 = 241.4... no
        # 9901 / 43 = 230.2... no
        # 9901 / 47 = 210.6... no
        # 9901 / 53 = 186.8... no
        # 9901 / 59 = 167.8... no
        # 9901 / 61 = 162.3... no
        # 9901 / 67 = 147.7... no
        # 9901 / 71 = 139.4... no
        # 9901 / 73 = 135.6... no
        # 9901 / 79 = 125.3... no
        # 9901 / 83 = 119.3... no
        # 9901 / 89 = 111.2... no
        # 9901 / 97 = 102.07... no
        # So 9901 is prime. LPF of 1000001 = 9901.
        lpf_np1 = 9901
        max_lpf = max(lpf_n, lpf_np1)
        if max_lpf < best_lpf:
            best_lpf = max_lpf
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
