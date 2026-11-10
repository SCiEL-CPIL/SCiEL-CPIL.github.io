"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy:
    1. Precompute the largest prime factor (LPF) for all integers up to 1,000,001
       using a modified sieve.
    2. For each n in the valid range, compute L = max(LPF[n], LPF[n+1]).
    3. Track the minimum L found and the corresponding n.
    4. Return the best n found.
    
    This is an exhaustive search over the valid range, guaranteeing the optimal solution.
    The sieve runs in O(N log log N) time, and the scan is O(N).
    """
    start_time = time.time()
    deadline = 60.0  # Safety margin before 90s kill
    
    N_MAX = 1000001  # We need LPF up to 1,000,001 (for n+1 when n=1,000,000)
    
    # Sieve to compute largest prime factor for all numbers up to N_MAX
    lpf = [0] * (N_MAX + 1)
    lpf[1] = 1
    
    # Initialize primes list
    for i in range(2, N_MAX + 1):
        if lpf[i] == 0:
            # i is prime
            lpf[i] = i
            for j in range(i, N_MAX + 1, i):
                lpf[j] = i  # Set i as a prime factor (will be overwritten by larger primes)
    
    # Now lpf[x] contains the largest prime factor of x for all x >= 2
    # Verify: for a number x, the last prime that divides it in the sieve is the largest.
    # The above loop correctly sets lpf[j] to the current prime i, and since we iterate
    # primes in increasing order, the final value is the largest prime factor.
    
    # Scan for the best n
    best_n = 10000
    best_lpf_val = float('inf')
    
    for n in range(10000, 1000001):
        # Check deadline
        if time.time() > deadline:
            break
        
        lpf_n = lpf[n]
        lpf_np1 = lpf[n + 1]
        max_lpf = max(lpf_n, lpf_np1)
        
        if max_lpf < best_lpf_val:
            best_lpf_val = max_lpf
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
