import math
import random

def is_prime(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    while True:
        c = random.randrange(1, n)
        f = lambda x: (x * x + c) % n
        x, y, d = 2, 2, 1
        while d == 1:
            x = f(x)
            y = f(f(y))
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def factorize(n):
    factors = {}
    while n > 1:
        if is_prime(n):
            factors[n] = factors.get(n, 0) + 1
            break
        d = pollard_rho(n)
        factors[d] = factors.get(d, 0) + 1
        n //= d
    return factors

def sigma_from_factors(factors):
    result = 1
    for p, exp in factors.items():
        result *= (p ** (exp + 1) - 1) // (p - 1)
    return result

def get_divisors(factors):
    divs = [1]
    for p, exp in factors.items():
        current_divs = []
        power = 1
        for _ in range(exp + 1):
            for d in divs:
                current_divs.append(d * power)
            power *= p
        divs = current_divs
    return divs

def construct():
    N = 5000000
    
    # 1. Sieve sigma
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # 2. Bucket M values
    from collections import defaultdict
    count = defaultdict(list)
    for k in range(1, N + 1):
        M_val = k * sigma[k]
        if M_val <= 10**18:
            count[M_val].append(k)
            
    # 3. Get top candidates
    # Sort by count descending
    candidates = sorted(count.items(), key=lambda item: len(item[1]), reverse=True)
    
    best_M = None
    best_k = []
    
    # Check top 50 candidates with exhaustive search
    for M_val, initial_ks in candidates[:50]:
        # Factorize M
        factors_M = factorize(M_val)
        
        # Generate all divisors d <= sqrt(M)
        # To do this efficiently, we can generate all divisors and filter,
        # or generate pairs. Given M <= 10^18, number of divisors is manageable.
        all_divs = get_divisors(factors_M)
        
        sqrt_M = math.isqrt(M_val)
        valid_ks = set()
        
        for d in all_divs:
            if d > sqrt_M:
                continue
            # Check if d * sigma(d) == M_val
            # We need sigma(d).
            # Since d divides M, we can get factors of d from factors_M.
            # But calculating factors of d for every d is O(num_divs * num_primes).
            # Alternative: 
            # We know d divides M. 
            # Let's just compute sigma(d) by factoring d? 
            # d is small (<= 10^9). Factoring 10^9 is fast.
            # Or we can derive factors from M's factors.
            
            # Efficient way:
            # Since d divides M, d = prod p^e where e <= exp_M(p).
            # We can iterate through divisors and compute sigma incrementally?
            # Simpler: Just factor d. d <= 10^9. 
            # Number of divisors is maybe 10k-100k.
            # Factoring 100k numbers of size 10^9 might be slow in Python.
            
            # Better: Pre-compute sigma for all divisors?
            # Let's stick to the derivation from M's factors.
            
            # Actually, for d <= sqrt(M), d is at most 10^9.
            # We can just factor d using the prime factors of M.
            # For each prime p in M, count how many times it divides d.
            
            factors_d = {}
            temp_d = d
            for p in factors_M.keys():
                if temp_d % p == 0:
                    exp = 0
                    while temp_d % p == 0:
                        temp_d //= p
                        exp += 1
                    factors_d[p] = exp
            
            # If temp_d > 1, d has a prime factor not in M? Impossible, d|M.
            # So factors_d is complete.
            
            sig_d = sigma_from_factors(factors_d)
            
            if d * sig_d == M_val:
                valid_ks.add(d)
                
        if len(valid_ks) > len(best_k):
            best_k = list(valid_ks)
            best_M = M_val
            
    if best_M is None:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_M, 'k': best_k}
