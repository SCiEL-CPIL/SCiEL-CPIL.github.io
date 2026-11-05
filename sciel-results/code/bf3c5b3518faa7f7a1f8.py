import math
import time

def construct():
    """
    Find a, b such that a + b = 123457 and max(P(a), P(b)) is minimized.
    P(n) is the largest prime factor of n.
    Uses binary search on the threshold K over primes, with a direct scan
    to verify feasibility at each K.
    """
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85  # seconds
    
    # Sieve of Eratosthenes to find all primes up to TARGET
    is_prime = [True] * (TARGET + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(TARGET**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, TARGET + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Precompute largest prime factor for all numbers up to TARGET
    lp = [0] * (TARGET + 1)
    lp[1] = 1  # P(1) = 1
    
    for p in primes:
        for multiple in range(p, TARGET + 1, p):
            lp[multiple] = p
    
    # Helper: check if there exist a, b >= 1 with a+b=TARGET and
    # lp[a] <= K and lp[b] <= K. Return the pair if found.
    def find_pair_for_K(K):
        """Find a,b with a+b=TARGET where both have largest prime factor <= K."""
        # a ranges from 1 to TARGET-1
        # We need lp[a] <= K and lp[TARGET-a] <= K
        for a in range(1, TARGET):
            if lp[a] <= K and lp[TARGET - a] <= K:
                return (a, TARGET - a)
        return None
    
    # Binary search on the index in primes for the smallest K that works
    # K must be at least 2 (smallest prime)
    lo = 0
    hi = len(primes) - 1
    best_pair = (1, TARGET - 1)
    best_K = lp[TARGET - 1]
    
    # First, find an upper bound: the brute-force optimum
    # We'll do a quick scan to find the actual minimum max_pf
    best_max_pf = float('inf')
    for a in range(1, TARGET):
        max_pf = max(lp[a], lp[TARGET - a])
        if max_pf < best_max_pf:
            best_max_pf = max_pf
            if time.time() - start_time > DEADLINE - 5:
                break
    
    # Now binary search: find smallest prime K such that a valid pair exists
    # The answer is the smallest K where find_pair_for_K(K) is not None
    # We know K is in [2, best_max_pf]
    
    # Find index range for primes in [2, best_max_pf]
    # lo: first prime >= 2, hi: first prime > best_max_pf
    lo_idx = 0  # primes[0] = 2
    hi_idx = len(primes) - 1
    # Shrink hi_idx to first prime > best_max_pf
    while hi_idx > 0 and primes[hi_idx] > best_max_pf:
        hi_idx -= 1
    
    # Binary search for the smallest prime K in primes[lo_idx:hi_idx+1]
    # such that find_pair_for_K(K) returns a valid pair
    result_pair = None
    left = lo_idx
    right = hi_idx
    while left <= right:
        if time.time() - start_time > DEADLINE - 3:
            break
        mid = (left + right) // 2
        K = primes[mid]
        pair = find_pair_for_K(K)
        if pair is not None:
            result_pair = pair
            right = mid - 1
        else:
            left = mid + 1
    
    if result_pair is not None:
        return [result_pair[0], result_pair[1]]
    
    # Fallback: return the best pair from the scan
    # Re-scan to get the actual pair
    best_a = 1
    best_b = TARGET - 1
    best_max_pf = float('inf')
    for a in range(1, TARGET):
        max_pf = max(lp[a], lp[TARGET - a])
        if max_pf < best_max_pf:
            best_max_pf = max_pf
            best_a = a
            best_b = TARGET - a
    
    return [best_a, best_b]
