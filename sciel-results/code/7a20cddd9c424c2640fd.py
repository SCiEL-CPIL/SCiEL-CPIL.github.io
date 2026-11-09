"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from collections import defaultdict

# --- Fast Factorization Utilities ---

def _is_prime(n):
    """Deterministic Miller-Rabin for n < 3.2 * 10^18"""
    if n < 2:
        return False
    # Small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    
    # Write n-1 as d * 2^r
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
        
    # Bases sufficient for n < 3.2 * 10^18
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def _pollard_rho(n):
    """Pollard's Rho factorization"""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    
    while True:
        x = random.randint(2, n - 1)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def _factorize(n):
    """Factorize n into prime factors. Returns dict {prime: exponent}."""
    if n <= 1:
        return {}
    factors = defaultdict(int)
    def _factor(n, factors):
        if n == 1:
            return
        if _is_prime(n):
            factors[n] += 1
            return
        d = _pollard_rho(n)
        _factor(d, factors)
        _factor(n // d, factors)
        
    _factor(n, factors)
    return dict(factors)

def _sigma_from_factors(factors):
    """Compute sigma(n) from its factorization dict."""
    result = 1
    for p, exp in factors.items():
        # sum_{i=0}^{exp} p^i
        # Using geometric series formula
        result *= (p**(exp + 1) - 1) // (p - 1)
    return result

def _sigma(n):
    """Compute sigma(n) - sum of divisors of n."""
    if n == 1:
        return 1
    f = _factorize(n)
    return _sigma_from_factors(f)

def _all_divisors_from_factors(factors):
    """Generate all divisors from factorization dict."""
    primes = list(factors.keys())
    exponents = [factors[p] for p in primes]
    
    divisors = [1]
    for i, p in enumerate(primes):
        current_divs = list(divisors)
        power = 1
        for e in range(1, exponents[i] + 1):
            power *= p
            for d in current_divs:
                divisors.append(d * power)
    return divisors

def _find_k_for_M(M, deadline):
    """Find all k such that k * sigma(k) = M and k <= sqrt(M)."""
    if M < 1:
        return []
    
    max_k = math.isqrt(M)
    if max_k < 1:
        return []
    
    # k must divide M? No. k * sigma(k) = M.
    # k doesn't have to divide M. For example, k=2, sigma(2)=3, M=6. 2 divides 6.
    # k=3, sigma(3)=4, M=12. 3 divides 12.
    # k=4, sigma(4)=7, M=28. 4 divides 28.
    # k=6, sigma(6)=12, M=72. 6 divides 72.
    # It seems k often divides M, but not necessarily always?
    # Actually, if k * sigma(k) = M, then k divides M.
    # Proof: M = k * sigma(k), so M / k = sigma(k), which is an integer.
    # So k MUST be a divisor of M.
    
    # Therefore, we only need to check divisors of M.
    # This is a HUGE optimization compared to checking all k <= sqrt(M).
    # The number of divisors of M is much smaller than sqrt(M).
    
    M_factors = _factorize(M)
    if not M_factors:
        return []
    
    divisors = _all_divisors_from_factors(M_factors)
    
    k_values = []
    for k in divisors:
        if time.time() > deadline:
            break
        if k > max_k:
            continue
        # Check if k * sigma(k) == M
        # We can compute sigma(k) efficiently if we know the factorization of k.
        # But k is a divisor of M, so its prime factors are a subset of M's prime factors.
        # We can compute sigma(k) directly from k's factorization.
        # Or just use _sigma(k).
        
        # Optimization: compute sigma(k) from the factorization of k.
        # Since k is a divisor of M, we can determine the exponent of each prime in k.
        # But it's easier to just factorize k. k is small (<= sqrt(M) <= 10^9).
        # _sigma(k) will factorize k.
        
        sig = _sigma(k)
        if k * sig == M:
            k_values.append(k)
            
    return k_values

def _search_collisions(max_k, deadline):
    """
    Search for M values with many k solutions by scanning k up to max_k.
    Returns dict M -> list of k values.
    """
    collisions = defaultdict(list)
    
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        
        # Compute sigma(k)
        # To speed up, we can use a sieve for small k.
        # But for now, let's just compute directly.
        # If max_k is large, this will be slow.
        # We should use a sieve.
        
        sig = _sigma(k)
        M = k * sig
        
        if M <= 10**18:
            collisions[M].append(k)
            
    return collisions

def _compute_sigma_sieve(n, deadline):
    """
    Compute sigma(k) for all k <= n using a sieve.
    Returns a list sigma[1..n] where sigma[i] is the sum of divisors of i.
    """
    sigma = [0] * (n + 1)
    sigma[1] = 1
    
    for i in range(1, n + 1):
        if time.time() > deadline:
            break
        # For each multiple of i, add i to sigma[multiple]
        for j in range(i, n + 1, i):
            sigma[j] += i
            
    return sigma

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (10s buffer for 120s limit)
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Strategy 1: Use sieve to compute sigma(k) for k up to some limit.
    # Then find M values with many k solutions.
    
    # How large can we go?
    # Sieve up to N takes O(N log N) time.
    # For N = 10^6, it's very fast.
    # For N = 10^7, it's a few seconds.
    # For N = 10^8, it might be too slow.
    
    # Let's try N = 10^6 first.
    N = 10**6
    
    sigma_arr = _compute_sigma_sieve(N, deadline)
    
    # Now scan k from 1 to N and find collisions.
    collisions = defaultdict(list)
    for k in range(1, N + 1):
        if time.time() > deadline:
            break
        
        sig = sigma_arr[k]
        M = k * sig
        
        if M <= 10**18:
            collisions[M].append(k)
            
    # Find the M with the most k values
    for M, ks in collisions.items():
        count = len(ks)
        if count > best_count:
            best_M = M
            best_ks = ks
            best_count = count
            
    # Strategy 2: If we have time, try larger N or search for specific M values.
    # But let's first see if the sieve approach finds a good M.
    
    # We can also try to extend the search by checking specific k values
    # that are likely to produce collisions.
    
    # For now, let's stick with the sieve result.
    
    # Ensure types are correct
    best_M = int(best_M)
    best_ks = [int(k) for k in best_ks]
    
    return {'M': best_M, 'k': best_ks}
# EVOLVE-BLOCK-END
