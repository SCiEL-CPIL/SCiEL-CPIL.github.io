import math
import time

def construct():
    """
    Find a, b such that a + b = 123457 and max(P(a), P(b)) is minimized.
    P(n) is the largest prime factor of n.
    """
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin

    # Precompute largest prime factor for all numbers 1..TARGET using a sieve.
    lp = [0] * (TARGET + 1)
    lp[1] = 1
    for i in range(2, TARGET + 1):
        if lp[i] == 0:
            # i is prime
            for j in range(i, TARGET + 1, i):
                lp[j] = i  # since i increases, the last prime that divides j is the largest

    # Now lp[n] is the largest prime factor of n for all n in 1..TARGET.
    # Binary search on threshold K: find smallest prime K such that
    # there exists a in [1, TARGET-1] with lp[a] <= K and lp[TARGET-a] <= K.
    
    # Collect all primes up to TARGET
    primes = []
    for i in range(2, TARGET + 1):
        if lp[i] == i:
            primes.append(i)
    
    # Binary search on the index in primes
    lo = 0
    hi = len(primes) - 1
    best_a = 1
    best_b = TARGET - 1
    best_val = float('inf')
    
    def feasible(K):
        """Check if there exists a in [1, TARGET-1] with lp[a] <= K and lp[TARGET-a] <= K."""
        for a in range(1, TARGET):
            if lp[a] <= K and lp[TARGET - a] <= K:
                return True, a
        return False, -1
    
    while lo <= hi:
        mid = (lo + hi) // 2
        K = primes[mid]
        ok, a_found = feasible(K)
        if ok:
            best_val = K
            best_a = a_found
            best_b = TARGET - a_found
            hi = mid - 1
        else:
            lo = mid + 1
        
        if time.time() - start_time > DEADLINE - 2:
            break
    
    # Fallback: if binary search didn't find anything (shouldn't happen), do linear scan
    if best_val == float('inf'):
        for a in range(1, TARGET):
            b = TARGET - a
            m = max(lp[a], lp[b])
            if m < best_val:
                best_val = m
                best_a = a
                best_b = b
            if time.time() - start_time > DEADLINE - 2:
                break
    
    return [best_a, best_b]
