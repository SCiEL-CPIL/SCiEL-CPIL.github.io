import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def _sigma(n):
    """Compute sum of divisors sigma(n) using trial division."""
    if n <= 0:
        return 0
    result = 1
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            p_power = 1
            p_sum = 1
            while temp % d == 0:
                temp //= d
                p_power *= d
                p_sum += p_power
            result *= p_sum
        d += 1 if d == 2 else 2  # skip even numbers after 2
    if temp > 1:
        result *= (1 + temp)
    return result


def _sigma_fast(n):
    """Faster sigma using prime trial division with 6k±1 optimization."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    result = 1
    temp = n
    # Factor out 2
    if temp % 2 == 0:
        p_power = 1
        p_sum = 1
        while temp % 2 == 0:
            temp //= 2
            p_power *= 2
            p_sum += p_power
        result *= p_sum
    # Factor out 3
    if temp % 3 == 0:
        p_power = 1
        p_sum = 1
        while temp % 3 == 0:
            temp //= 3
            p_power *= 3
            p_sum += p_power
        result *= p_sum
    # Try 6k±1
    d = 5
    step = 2
    while d * d <= temp:
        if temp % d == 0:
            p_power = 1
            p_sum = 1
            while temp % d == 0:
                temp //= d
                p_power *= d
                p_sum += p_power
            result *= p_sum
        d += step
        step = 6 - step
    if temp > 1:
        result *= (1 + temp)
    return result


def construct():
    start_time = time.time()
    deadline = 130.0  # 10s safety margin
    
    # Strategy: Search for M with many k such that k*sigma(k) = M
    # and k <= sqrt(M).
    # 
    # Key insight: For k with many small prime factors, sigma(k)/k can be 
    # "large" in a controlled way. We want many k_i to produce the same M.
    #
    # Approach: Enumerate k values (compositions of small primes), compute
    # M = k * sigma(k), and group by M. Then pick the M with most k's.
    #
    # To get many k's with the same M, we need k*sigma(k) to be constant.
    # This is rare, but we can search systematically.
    
    # Method 1: Direct enumeration of k up to some limit, group by M
    # Method 2: For specific M candidates, find all k with k*sigma(k)=M
    
    # Let's try a hybrid: enumerate k's that are products of small primes,
    # compute M = k*sigma(k), group by M, find best M.
    
    # We need k <= sqrt(M) = sqrt(k*sigma(k)), so we need k <= k*sigma(k)
    # which means sigma(k) >= 1, always true. More precisely:
    # k <= sqrt(k*sigma(k)) => k^2 <= k*sigma(k) => k <= sigma(k)
    # This is always true since sigma(k) >= k+1 for k >= 1 (except k=1 where sigma(1)=1).
    # Wait: for k=1, sigma(1)=1, so k=1 <= sqrt(1) = 1. OK.
    # For k>=2, sigma(k) >= k+1 > k, so k < sigma(k), hence k < sqrt(k*sigma(k)).
    # So all k >= 1 satisfy k <= sqrt(k*sigma(k)).
    
    # So the constraint k <= sqrt(M) is automatically satisfied!
    # We just need to find M with many distinct k such that k*sigma(k) = M.
    
    # Enumerate k values and group by M = k*sigma(k)
    # Limit k to reasonable range to stay within time.
    
    # Precompute sigma for k up to some limit
    # Use sieve-like approach for efficiency
    
    max_k = 200000  # will adjust based on time
    sigma_arr = [0] * (max_k + 1)
    for i in range(1, max_k + 1):
        sigma_arr[i] = 1  # each number contributes at least 1 (itself... wait no)
        # Actually sigma(n) = sum of divisors. Let me use a different sieve.
    
    # Better: sieve for sigma
    sigma_arr = [0] * (max_k + 1)
    for d in range(1, max_k + 1):
        for m in range(d, max_k + 1, d):
            sigma_arr[m] += d
    
    # Now group by M = k * sigma(k)
    M_to_ks = defaultdict(list)
    for k in range(1, max_k + 1):
        M = k * sigma_arr[k]
        if M <= 10**18:
            M_to_ks[M].append(k)
    
    # Find M with most k's
    best_M = 1
    best_ks = [1]
    for M, ks in M_to_ks.items():
        if len(ks) > len(best_ks):
            best_M = M
            best_ks = ks
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # Try larger max_k
        for max_k2 in [500000, 1000000, 2000000, 5000000, 10000000]:
            if time.time() - start_time > deadline - 10:
                break
            sigma_arr2 = [0] * (max_k2 + 1)
            for d in range(1, max_k2 + 1):
                for m in range(d, max_k2 + 1, d):
                    sigma_arr2[m] += d
            
            # Only need to check new k's from max_k+1 to max_k2
            # But M_to_ks already has entries; we can extend
            for k in range(max_k + 1, max_k2 + 1):
                M = k * sigma_arr2[k]
                if M <= 10**18:
                    M_to_ks[M].append(k)
            
            # Update best
            for M, ks in M_to_ks.items():
                if len(ks) > len(best_ks):
                    best_M = M
                    best_ks = ks
            
            max_k = max_k2
    
    # Ensure best_ks are distinct and valid
    best_ks = list(set(best_ks))
    best_ks.sort()
    
    # Verify each k satisfies k*sigma(k) = best_M
    verified_ks = []
    for k in best_ks:
        if k * _sigma_fast(k) == best_M and k <= math.isqrt(best_M):
            verified_ks.append(k)
    
    if not verified_ks:
        # Fallback to known solution
        verified_ks = [336, 372, 434]
        best_M = 333312
    
    return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
# EVOLVE-BLOCK-END
