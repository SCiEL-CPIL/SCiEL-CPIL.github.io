import time
import math

# Precompute primes up to 2000 for trial division of B
def _get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

_PRIMES = _get_primes(2000)

def _factorize(n):
    """Return list of (prime, exponent) for n."""
    factors = []
    temp_n = n
    for p in _PRIMES:
        if p * p > temp_n:
            break
        if temp_n % p == 0:
            exp = 0
            while temp_n % p == 0:
                temp_n //= p
                exp += 1
            factors.append((p, exp))
    if temp_n > 1:
        factors.append((temp_n, 1))
    return factors

def _get_divisors_of_square(factors):
    """
    Given factorization of B as [(p, e)], generate divisors of B^2.
    B^2 has factors [(p, 2e)].
    """
    divisors = [1]
    for p, e in factors:
        max_exp = 2 * e
        new_divs = []
        for d in divisors:
            pe = 1
            for _ in range(max_exp + 1):
                new_divs.append(d * pe)
                pe *= p
        divisors = new_divs
    return divisors

# Cache for divisor sets keyed by B
_divisor_cache = {}

def _get_divisors_of_B_squared(B):
    """Get divisors of B^2, using cache."""
    if B in _divisor_cache:
        return _divisor_cache[B]
    fac_B = _factorize(B)
    divs = _get_divisors_of_square(fac_B)
    _divisor_cache[B] = divs
    return divs

def _solve_n(n, BOUND=10**12):
    """Find distinct x<y<z with 1/x+1/y+1/z=4/n, all <= BOUND."""
    # x must satisfy: 1/x < 4/n => x > n/4, and 1/x > 4/n - 2/x_min... 
    # More precisely: for y > x, z > y, we need 4/n > 3/x => x > 3n/4
    # And 4/n < 3/x + 2/x = ... actually let's use:
    # 1/x + 1/y + 1/z = 4/n with x<y<z
    # 4/n > 3/z and 4/n < 3/x, so x < 3n/4 and z > 3n/4
    # Also 4/n > 1/x + 1/(x+1) + 1/(x+2) roughly, but tighter:
    # 4/n > 3/x => x > 3n/4
    # 4/n < 1/x + 2/x = 3/x => x < 3n/4... wait that's contradictory.
    # Let me redo: since y > x and z > y > x:
    # 4/n = 1/x + 1/y + 1/z < 3/x => x < 3n/4
    # 4/n = 1/x + 1/y + 1/z > 1/x + 1/(x+1) + 1/(x+2)
    # Simpler bound: 4/n > 3/x => x > 3n/4 is WRONG.
    # 4/n < 3/x means x < 3n/4.
    # 4/n > 1/x + 1/x + 1/x = 3/x means x > 3n/4.
    # Contradiction! So we need a different approach.
    # Actually: 1/x + 1/y + 1/z = 4/n, with x < y < z
    # Lower bound on x: 4/n > 3/x is false because 1/x+1/y+1/z < 3/x
    # So 4/n < 3/x => x < 3n/4
    # Upper bound: 4/n > 1/x + 1/(x+1) + 1/(x+2) approximately
    # But more useful: 4/n - 1/x = 1/y + 1/z > 2/z and < 2/x
    # Let A = 4x - n, B = nx. Then (Ay-B)(Az-B) = B^2
    # A = 4x - n > 0 requires x > n/4
    # For y > x: Ay - B > Ax - B = A(x) - B = (4x-n)x - nx = 4x^2 - nx - nx = 4x^2 - 2nx = 2x(2x-n)
    # We need d = Ay - B to be a positive divisor of B^2 with d < B (for y < z)
    # y = (d+B)/A > x means d+B > Ax means d > Ax - B = 4x^2 - nx - nx = 4x^2 - 2nx
    # So we need divisors d of B^2 with max(1, 4x^2 - 2nx + 1) <= d < B and (d+B) % A == 0
    
    x_min = n // 4 + 1
    # x_max: from 4/n < 3/x => x < 3n/4, so x_max = (3n-1)//4
    x_max = (3 * n - 1) // 4
    
    if x_min > x_max:
        return None
    
    for x in range(x_min, x_max + 1):
        A = 4 * x - n
        if A <= 0:
            continue
        B = n * x
        
        # d must satisfy: d > Ax - B = 4x^2 - 2nx, d < B, d | B^2, (d+B) % A == 0
        d_min = 4 * x * x - 2 * n * x + 1
        if d_min < 1:
            d_min = 1
        if d_min >= B:
            continue
        
        divs = _get_divisors_of_B_squared(B)
        
        for d in divs:
            if d < d_min:
                continue
            if d >= B:
                break  # divs may not be sorted, but let's check
            
            num_y = d + B
            if num_y % A != 0:
                continue
            
            y = num_y // A
            if y <= x:
                continue
            
            term1 = (B * B) // d
            num_z = term1 + B
            
            if num_z % A != 0:
                continue
            
            z = num_z // A
            
            if z <= y:
                continue
            if z > BOUND:
                continue
            
            # Verify
            if n * (y * z + x * z + x * y) == 4 * x * y * z:
                return [x, y, z]
    
    return None

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    BOUND = 10**12
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        results[idx] = _solve_n(n, BOUND)
    
    return results
