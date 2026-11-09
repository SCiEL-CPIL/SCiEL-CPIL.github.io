import time
import math

# Precompute primes up to 100000
def _sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    return primes

PRIMES = _sieve(100000)

def _factorize(n):
    """Return a list of (prime, exponent) for n."""
    factors = []
    for p in PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            factors.append((p, exp))
    if n > 1:
        factors.append((n, 1))
    return factors

# Cache for factorizations to avoid recomputation
_factorize_cache = {}

def _factorize_cached(n):
    """Factorize with caching."""
    if n in _factorize_cache:
        return _factorize_cache[n]
    result = _factorize(n)
    _factorize_cache[n] = result
    return result

def _divisors_from_factors(factors):
    """Generate all divisors from a list of (prime, exponent)."""
    divs = [1]
    for p, e in factors:
        current_divs = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                current_divs.append(d * pe)
            pe *= p
        divs = current_divs
    return divs

def construct():
    deadline = time.time() + 88
    answers = []
    total_n = 254
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x values in a smarter order: start from values likely to yield solutions
        # x close to n/4 gives larger a, which may constrain better
        # But also try small x first since they're more likely to have solutions
        x_order = list(range(x_min, x_max + 1))
        
        for x in x_order:
            if time.time() > deadline:
                break
            if found:
                break
            
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            g = math.gcd(a, b)
            a_red = a // g
            b_red = b // g
            
            b_red_sq = b_red * b_red
            
            # Factorize b_red_sq
            factors_b = _factorize(b_red)
            factors_sq = [(p, 2*e) for p, e in factors_b]
            
            divs = _divisors_from_factors(factors_sq)
            
            for d in divs:
                if time.time() > deadline:
                    break
                if found:
                    break
                
                num_y = b_red + d
                if num_y % a_red != 0:
                    continue
                y = num_y // a_red
                
                if y <= x:
                    continue
                
                num_z = b_red + b_red_sq // d
                if num_z % a_red != 0:
                    continue
                z = num_z // a_red
                
                if z <= y:
                    continue
                
                if y > 10**12 or z > 10**12:
                    continue
                
                # Verify
                lhs_num = n * y * z + n * x * z + n * x * y
                rhs_num = 4 * x * y * z
                if lhs_num == rhs_num:
                    found = [x, y, z]
                    break
        
        answers.append(found)
    
    return answers
