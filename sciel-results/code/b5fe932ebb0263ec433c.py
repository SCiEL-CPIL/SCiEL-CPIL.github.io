import time
import math
from itertools import combinations

def _tau_perp(n):
    """Count adjacent divisor pairs with gcd=1."""
    if n < 2:
        return 0
    # Factorize n
    divisors = set([1, n])
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    divs = sorted(divisors)
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit+1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    primes = _sieve(100)
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    best_score = _tau_perp(best_n)
    
    # Try subsets of primes, size 8 to 15
    for k in range(8, 16):
        if time.time() > deadline:
            break
        for combo in combinations(primes, k):
            if time.time() > deadline:
                break
            prod = 1
            for p in combo:
                prod *= p
                if prod > 10**18:
                    break
            if prod > 10**18 or prod < 2:
                continue
            score = _tau_perp(prod)
            if score > best_score:
                best_score = score
                best_n = prod
    
    return best_n
