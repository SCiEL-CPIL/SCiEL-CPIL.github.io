import time
import math
from math import isqrt

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n exactly."""
    if n < 2:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            # sigma(p^exp) = 1 + p + ... + p^exp
            s = (p ** (exp + 1) - 1) // (p - 1)
            result *= s
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    # The known solution: 336*sigma(336) = 372*sigma(372) = 434*sigma(434) = 333312
    # Let's search for M values with many solutions k where k * sigma(k) = M
    # Strategy: enumerate k up to some limit, compute M = k*sigma(k), count collisions.
    # For each k, we need k <= sqrt(M) = sqrt(k*sigma(k)), so sigma(k) >= k, which is always true.
    
    # Better strategy: we want to find M with many k such that k*sigma(k) = M.
    # Let's try a larger search space.
    
    # First, let's use the known answer as fallback
    best_M = 333312
    best_ks = [336, 372, 434]
    
    # Search: for each k in range, compute M = k * sigma(k), track which M has many k's
    # But we need k <= sqrt(M) = sqrt(k*sigma(k)), so sigma(k) >= k
    # sigma(k) >= k+1 for k >= 2, and sigma(1) = 1, so k=1 gives M=1, k <= 1, ok.
    
    # Let's do a focused search. We'll iterate k and build a dict M -> list of k's.
    # But M can be up to 10^18, so we can't just use all k.
    # Key insight: we need k <= sqrt(M) = sqrt(k * sigma(k)), so k^2 <= k * sigma(k), i.e., k <= sigma(k).
    # This is always satisfied for k >= 1.
    
    # The constraint is M <= 10^18, so k * sigma(k) <= 10^18.
    # For k up to ~10^9, sigma(k) can be up to ~k * log(log(k)), so k*sigma(k) ~ k^2 * log(log(k)).
    # k^2 * log(log(k)) <= 10^18 => k <= ~10^9 / sqrt(log(log(k)))
    # Let's try k up to a few million first.
    
    # Build collision map
    collision_map = {}
    
    # Phase 1: scan k from 1 to some limit
    # For speed, let's try up to 500,000 first, then extend if time permits.
    
    k_limit = 1_000_000
    for k in range(1, k_limit + 1):
        if time.time() > deadline:
            break
        s = _sigma(k)
        M = k * s
        if M > 10**18:
            continue
        # Check k <= sqrt(M)
        if k * k <= M:  # equivalent to k <= sqrt(M)
            if M not in collision_map:
                collision_map[M] = []
            collision_map[M].append(k)
    
    # Find best M (most solutions)
    current_best_count = len(best_ks)
    for M, ks in collision_map.items():
        if len(ks) > current_best_count:
            current_best_count = len(ks)
            best_M = M
            best_ks = ks
    
    # If we found something better, use it
    if current_best_count > 3:
        # Verify all solutions
        valid_ks = []
        for k in best_ks:
            if k * _sigma(k) == best_M and k * k <= best_M and 1 <= k <= isqrt(best_M):
                valid_ks.append(k)
        if len(valid_ks) > current_best_count:
            current_best_count = len(valid_ks)
            best_ks = valid_ks
        else:
            best_ks = valid_ks
            current_best_count = len(valid_ks)
    
    # Ensure types are correct
    best_M = int(best_M)
    best_ks = [int(k) for k in best_ks]
    # Deduplicate and sort
    best_ks = sorted(set(best_ks))
    
    # Final validation
    final_ks = []
    for k in best_ks:
        if isinstance(k, int) and 1 <= k <= isqrt(best_M) and k * _sigma(k) == best_M:
            final_ks.append(k)
    
    if len(final_ks) >= 1:
        return {'M': best_M, 'k': final_ks}
    
    # Fallback to known solution
    return {'M': 333312, 'k': [336, 372, 434]}
