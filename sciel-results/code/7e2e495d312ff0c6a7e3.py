import time
from math import gcd

# Precompute small primes up to 100000 for fast factorization
def _sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

_PRIMES = _sieve(100000)

def _factorize(n):
    """Return dict of prime -> exponent for n using precomputed primes."""
    factors = {}
    for p in _PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization dict."""
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return divs

def _find_solution_for_n(n, deadline):
    """Find distinct positive integer denominators x,y,z with 1/x+1/y+1/z=4/n."""
    # x must be > n/4
    x_min = n // 4 + 1
    # x must be < 3n/4 (since 1/x < 4/n - 1/y - 1/z < 4/n, and y,z > x implies...)
    # Actually, 1/x + 1/y + 1/z = 4/n.
    # If x >= 3n/4, 1/x <= 4/3n. 1/y + 1/z <= 4/3n.
    # y > x => 1/y < 4/3n.
    # It's safe to scan up to 3n/4.
    x_max = (3 * n) // 4
    
    for x in range(x_min, x_max + 1):
        if time.time() > deadline:
            return None
        A = 4 * n - x
        B = n * x
        if A <= 0:
            continue
        
        # Factorize B to get divisors of B^2
        fb = _factorize(B)
        fb2 = {p: 2 * e for p, e in fb.items()}
        
        # Check number of divisors to avoid explosion (unlikely for B < 50000)
        num_divs = 1
        for e in fb2.values():
            num_divs *= (e + 1)
            if num_divs > 100000:
                # Fallback to y-scan if too many divisors? Or just skip?
                # For n<=256, B is small, so this shouldn't happen.
                pass

        divs = _divisors_from_factors(fb2)
        
        # We need u < B and u | B^2
        # u = A*y - B
        # y = (u+B)/A
        # z = (v+B)/A where v = B^2/u
        # Conditions:
        # 1. u < B
        # 2. u % A == (-B) % A
        # 3. y > x
        # 4. z > y
        # 5. z <= 10^12
        
        for u in divs:
            if u >= B:
                continue
            if u % A != (-B) % A:
                continue
            
            y = (u + B) // A
            if y <= x:
                continue
            
            v = (B * B) // u
            z = (v + B) // A
            
            if z <= y:
                continue
            if z > 10**12:
                continue
            
            # Verify solution
            # 1/x + 1/y + 1/z == 4/n
            # (yz + xz + xy) / xyz == 4n / xyz ? No.
            # n(yz + xz + xy) == 4xyz
            if n * (y * z + x * z + x * y) == 4 * x * y * z:
                return [x, y, z]
                
    return None

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        results[idx] = _find_solution_for_n(n, deadline)
    return results
