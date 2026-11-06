import math
import time
from itertools import product

# EVOLVE-BLOCK-START

def _tau_perp(n):
    """Compute the number of adjacent coprime divisor pairs for n."""
    if n < 2:
        return 0
    # Factorize n
    factors = {}
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            factors[p] = cnt
        p += 1 if p == 2 else 2
    if temp > 1:
        factors[temp] = 1
    
    # Generate all divisors
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        for d in divs:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divs = new_divs
    divs.sort()
    
    # Count adjacent coprime pairs
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Product of first 15 primes (47#)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    best_n = 1
    for p in primes:
        best_n *= p
    best_score = _tau_perp(best_n)
    
    # Try replacing the largest prime with a larger prime
    # Product of first 14 primes
    prod14 = 1
    for p in primes[:14]:
        prod14 *= p
    
    # We can replace 47 with any prime p such that prod14 * p <= 10^18
    max_p = 10**18 // prod14
    
    # Find all primes in range (47, max_p]
    # Generate primes up to max_p
    # max_p is about 10^18 / 9699690 = ~1.03e14, too large to sieve naively
    # But we only need to check if prod14 * p is valid, and we want to try
    # various primes to see if they improve the score.
    
    # Actually, let's try a few strategic replacements
    # Replace 47 with the largest prime <= max_p
    # Or try primes that might create better adjacencies
    
    # Let's try replacing 47 with primes near max_p and some others
    # First, let's try the maximum possible prime
    # We need a primality test for numbers up to ~10^14
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Try the largest prime <= max_p
    p_max = max_p
    while p_max > 47 and not is_prime(p_max):
        p_max -= 2 if p_max % 2 == 0 else 1  # Skip even numbers mostly
    
    candidates = []
    if p_max > 47:
        candidates.append(p_max)
    
    # Try some other primes: replace 47 with primes in geometric progression
    # and some small primes that might create better patterns
    for p in [47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        if prod14 * p <= 10**18:
            candidates.append(p)
    
    # Try larger primes in log-spaced intervals
    log_min = math.log(47)
    log_max = math.log(max_p)
    for i in range(20):
        p_candidate = int(math.exp(log_min + (log_max - log_min) * i / 20))
        # Find nearest prime
        if p_candidate > 47:
            p_check = p_candidate
            while p_check > 47 and not is_prime(p_check):
                p_check -= 1
            if p_check > 47 and prod14 * p_check <= 10**18:
                candidates.append(p_check)
    
    candidates = list(set(candidates))
    
    for p in candidates:
        if time.time() > deadline:
            break
        n_candidate = prod14 * p
        if 2 <= n_candidate <= 10**18:
            score = _tau_perp(n_candidate)
            if score > best_score:
                best_score = score
                best_n = n_candidate
    
    return best_n

# EVOLVE-BLOCK-END
