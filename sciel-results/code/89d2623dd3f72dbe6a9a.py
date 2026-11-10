import math
from math import isqrt

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n using trial division."""
    if n == 1:
        return 1
    result = 1
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            power_sum = 1
            power = 1
            while temp % d == 0:
                temp //= d
                power *= d
                power_sum += power
            result *= power_sum
        d += 1
    if temp > 1:
        result *= (1 + temp)
    return result

def _sigma_from_factors(factors):
    """Compute sigma from prime factorization {p: exponent}."""
    result = 1
    for p, a in factors.items():
        result *= (p**(a+1) - 1) // (p - 1)
    return result

def construct():
    # Strategy: Find M with many k such that k*sigma(k) = M
    # 
    # Key insight: If we can find many k with the same ratio sigma(k)/k = r,
    # then M = k*r works for all such k.
    #
    # sigma(k)/k = prod over primes p|k of (1 + 1/p + ... + 1/p^(a_p-1))
    #
    # For k = product of distinct primes p_1...p_m (squarefree):
    # sigma(k)/k = prod (1 + 1/p_i)
    #
    # So if we find many squarefree k with the same set of prime factors (just different... wait, no)
    # Actually for squarefree k, sigma(k)/k depends on the SET of primes.
    #
    # Better approach: Use k = p^a where different (p,a) give same sigma(p^a)/p^a
    # sigma(p^a)/p^a = (p^(a+1)-1)/(p-1)/p^a = (p - p^(-a))/(p-1)
    #
    # Actually, let's think differently. We want k*sigma(k) = M.
    # Let's try to construct M as a product and find k's that work.
    #
    # Known identity: For k = 2^a * 3^b, sigma(k) = sigma(2^a)*sigma(3^b)
    # k*sigma(k) = 2^a*3^b * (2^(a+1)-1)/(2-1) * (3^(b+1)-1)/(3-1)
    #            = 2^a*3^b * (2^(a+1)-1) * (3^(b+1)-1)/2
    #
    # Let's try a different approach: search for M values that have multiple k solutions.
    #
    # Approach: Generate candidate M values by computing k*sigma(k) for various k,
    # then find M values that appear multiple times.
    #
    # But we need M <= 10^18 and k <= sqrt(M), so k*sigma(k) = M means sigma(k) = M/k >= sqrt(M).
    # Since sigma(k) >= k+1 > k for k > 1, we need k*sigma(k) where sigma(k) >= k+1.
    #
    # Let me try to find many k with the same k*sigma(k) value.
    #
    # For k = 2^a: k*sigma(k) = 2^a * (2^(a+1)-1)
    # For k = 3^b: k*sigma(k) = 3^b * (3^(b+1)-1)/2
    #
    # Let's compute k*sigma(k) for many k and look for collisions.
    #
    # Actually, a better strategy: use the fact that if M = n * sigma(n) for multiple n,
    # we need to find such M. Let's search systematically.
    
    import time
    start_time = time.time()
    deadline = start_time + 100  # 100 second deadline with safety margin
    
    # First, let's try a known good example and then search for better ones
    # Known: M = 333312 has k = 336, 372, 434 (3 solutions)
    
    # Let's search for M with more solutions by computing k*sigma(k) for k up to some limit
    # and finding values that appear most frequently.
    
    # We need k <= sqrt(M) and M = k*sigma(k) <= 10^18
    # So k*sigma(k) <= 10^18, and since sigma(k) >= k+1, k*(k+1) <= 10^18
    # So k < ~10^9
    
    # Let's compute k*sigma(k) for k in a range and find collisions.
    # To be efficient, let's focus on k with small prime factors.
    
    # Generate k values and compute k*sigma(k)
    # We'll use a dictionary to count occurrences
    
    # Strategy: For each k, compute v = k*sigma(k). If v <= 10^18, record it.
    # Then find v with the most k's.
    
    # To make this efficient, let's generate k's with at most a few distinct prime factors.
    
    # Let's try: for each "type" of k (defined by prime factorization pattern),
    # compute k*sigma(k) and see if we get collisions.
    
    # Actually, let's just do a direct search for k up to a reasonable limit.
    # sigma(k) can be computed quickly for k up to ~10^7 or so.
    
    # Let's use a sieve-like approach to compute sigma up to N.
    N = 500000  # Compute sigma for k up to N
    sigma_arr = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma_arr[j] += i
    
    # Now compute k*sigma(k) for k up to N and find collisions
    from collections import defaultdict
    k_by_M = defaultdict(list)
    
    for k in range(2, N + 1):
        v = k * sigma_arr[k]
        if v <= 10**18:
            k_by_M[v].append(k)
    
    # Find M with most solutions
    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3
    
    for M, k_list in k_by_M.items():
        if len(k_list) > best_count:
            best_count = len(k_list)
            best_M = M
            best_k_list = k_list
            if best_count >= 50:  # Good enough, stop early
                break
    
    # Check time
    if time.time() > deadline:
        return {'M': best_M, 'k': best_k_list}
    
    # Now let's try to extend by searching for more k for the best M found
    # For the best M, we need k*sigma(k) = M, so sigma(k) = M/k
    # k must divide M, and k <= sqrt(M)
    
    # Let's also try a broader search with larger k using direct sigma computation
    # for k that are divisors of candidate M values
    
    # Get top candidates
    top_candidates = sorted(k_by_M.items(), key=lambda x: len(x[1]), reverse=True)[:20]
    
    for M, k_list in top_candidates:
        if time.time() > deadline:
            break
        # Try to find more k for this M
        # k must satisfy: k | M, k <= sqrt(M), and sigma(k) = M/k
        sqrtM = isqrt(M)
        existing = set(k_list)
        
        # Find divisors of M up to sqrtM
        # M can be large, so let's be careful
        if M > 10**15:
            continue  # Too large to factorize easily
        
        # Factorize M to find divisors
        temp = M
        factors = {}
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        
        # Generate all divisors up to sqrtM
        def gen_divisors(factors, current=1, idx=0):
            if idx == len(factors):
                if current <= sqrtM:
                    yield current
                return
            p, e = list(factors.items())[idx]
            for i in range(e + 1):
                yield from gen_divisors(factors, current * (p ** i), idx + 1)
        
        found_more = False
        for k in gen_divisors(factors):
            if k in existing or k < 2:
                continue
            if k > sqrtM:
                continue
            if M % k != 0:
                continue
            target_sigma = M // k
            # Check if sigma(k) == target_sigma
            if _sigma(k) == target_sigma:
                k_list.append(k)
                found_more = True
        
        if found_more and len(k_list) > best_count:
            best_count = len(k_list)
            best_M = M
            best_k_list = k_list
    
    # Final check
    # Verify our answer
    if best_M > 10**18:
        best_M = 333312
        best_k_list = [336, 372, 434]
    
    # Ensure k values are valid
    valid_k = []
    sqrtM = isqrt(best_M)
    for k in best_k_list:
        if 1 <= k <= sqrtM and _sigma(k) * k == best_M:
            valid_k.append(k)
    
    # Remove duplicates and sort
    valid_k = sorted(set(valid_k))
    
    if not valid_k:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': best_M, 'k': valid_k}
