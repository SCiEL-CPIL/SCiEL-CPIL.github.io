import math
from collections import defaultdict

def _sigma(n):
    """Compute sigma(n) by trial division up to sqrt(n)."""
    if n == 1:
        return 1
    result = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            p_sum = 1
            p_pow = 1
            while n % d == 0:
                n //= d
                p_pow *= d
                p_sum += p_pow
            result *= p_sum
        d += 1
    if n > 1:
        result *= (1 + n)
    return result

def construct():
    import time
    start_time = time.time()
    DEADLINE = 115.0  # seconds, leaving 5s safety margin

    best_result = {'M': 333312, 'k': [336, 372, 434]}
    best_count = 3

    def check_and_update(M, ks):
        nonlocal best_result, best_count
        if len(ks) > best_count:
            best_count = len(ks)
            best_result = {'M': int(M), 'k': [int(x) for x in ks[:1000]]}

    # Phase 1: Fast numpy sieve for broad k-range scan
    try:
        import numpy as np
        use_numpy = True
    except ImportError:
        use_numpy = False

    if use_numpy:
        # Use a moderate max_k that fits in memory and time
        max_k = 5000000  # 5 million - fits well in memory
        sigma = np.zeros(max_k + 1, dtype=np.int64)
        
        # Sieve for sigma
        for i in range(1, max_k + 1):
            sigma[i::i] += i
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            return best_result

        k_arr = np.arange(1, max_k + 1, dtype=np.int64)
        M_arr = k_arr * sigma[1:max_k + 1]
        valid = (M_arr <= 10**18) & (k_arr <= sigma[1:max_k + 1])
        
        valid_idx = np.where(valid)[0]
        if len(valid_idx) > 0:
            M_valid = M_arr[valid_idx]
            k_valid = k_arr[valid_idx]
            
            unique_M, inverse, counts = np.unique(M_valid, return_inverse=True, return_counts=True)
            
            # Find top M values by count
            top_n = min(100, len(unique_M))
            top_indices = np.argsort(counts)[::-1][:top_n]
            for idx in top_indices:
                cnt = int(counts[idx])
                if cnt > best_count:
                    best_M = int(unique_M[idx])
                    mask = inverse == idx
                    ks = k_valid[mask].tolist()
                    check_and_update(best_M, ks)

        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            return best_result

    # Phase 2: Targeted search for M values with many divisors
    # Generate candidate M values that are products of small primes with various exponents
    # These are likely to have many k solutions
    
    def generate_candidate_Ms():
        """Generate highly composite-like numbers M = prod(p_i^a_i) where M <= 10^18."""
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        candidates = set()
        
        # Generate M as product of first few primes with varying exponents
        def _gen(idx, current_M, exponents):
            if current_M > 10**18:
                return
            if idx == len(small_primes):
                if current_M > 1:
                    candidates.add(current_M)
                return
            p = small_primes[idx]
            # Try exponents 0 through max such that p^a doesn't exceed limit
            val = 1
            for a in range(0, 60):
                if current_M * val > 10**18:
                    break
                exponents.append(a)
                _gen(idx + 1, current_M * val, exponents)
                exponents.pop()
                if val > 10**18 // p:
                    break
                val *= p
        
        _gen(0, 1, [])
        return list(candidates)

    # Use a smarter approach: focus on M values that are multiples of highly composite numbers
    # and check if they can be written as k*sigma(k) for many k
    
    # First, let's try specific known-good M values and their multiples
    # M = 333312 = 2^6 * 3^2 * 7 * 13 * 17 has 3 solutions
    # Let's explore M values with more divisors
    
    # Generate candidates: M = lcm-based or highly composite
    # For each candidate M, we need to find all k such that k*sigma(k) = M
    # This means k | M and sigma(k) = M/k
    
    # Strategy: For a given M, iterate over divisors k of M and check if sigma(k) = M/k
    # But we need k <= sqrt(M)
    
    # Let's generate a set of promising M values
    promising_Ms = set()
    
    # Start with the known best
    promising_Ms.add(333312)
    
    # Generate M values as products of small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    
    def gen_M(idx, current_M):
        if current_M > 10**18:
            return
        if idx == len(small_primes):
            if current_M > 1 and current_M <= 10**18:
                promising_Ms.add(current_M)
            return
        p = small_primes[idx]
        val = 1
        for _ in range(0, 60):
            if current_M * val > 10**18:
                break
            gen_M(idx + 1, current_M * val)
            if val > 10**18 // p:
                break
            val *= p
    
    # Limit generation to keep it manageable
    # Only use first 6 primes with limited exponents
    def gen_M_limited(idx, current_M, max_primes=6, max_exp=8):
        if current_M > 10**18:
            return
        if idx == max_primes:
            if current_M > 1 and current_M <= 10**18:
                promising_Ms.add(current_M)
            return
        p = small_primes[idx]
        val = 1
        for _ in range(0, max_exp + 1):
            if current_M * val > 10**18:
                break
            gen_M_limited(idx + 1, current_M * val, max_primes, max_exp)
            if val > 10**18 // p:
                break
            val *= p
    
    gen_M_limited(0, 1, 6, 8)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_result
    
    # For each promising M, find all valid k
    # k must be a divisor of M, k <= sqrt(M), and sigma(k) = M/k
    for M in sorted(promising_Ms, reverse=True):  # Try larger M first (more room for k)
        if time.time() - start_time > DEADLINE:
            break
        
        sqrt_M = math.isqrt(M)
        ks = []
        
        # Find all divisors of M up to sqrt_M
        # Factorize M first
        n = M
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        
        # Generate all divisors of M up to sqrt_M
        def gen_divisors(idx, current, prime_list):
            if current > sqrt_M:
                return
            if idx == len(prime_list):
                ks.append(current)
                return
            p, e = prime_list[idx]
            val = 1
            for _ in range(e + 1):
                if current * val > sqrt_M:
                    break
                gen_divisors(idx + 1, current * val, prime_list)
                val *= p
        
        prime_list = list(factors.items())
        gen_divisors(0, 1, prime_list)
        
        # Check each divisor k
        valid_ks = []
        for k in ks:
            if k <= sqrt_M and M % k == 0:
                if _sigma(k) == M // k:
                    valid_ks.append(k)
        
        if len(valid_ks) > best_count:
            check_and_update(M, valid_ks)
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            break

    return best_result
