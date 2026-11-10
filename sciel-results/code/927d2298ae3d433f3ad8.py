import time
from math import gcd

# EVOLVE-BLOCK-START
def _factorize(n):
    """Return dict of prime -> exponent for n."""
    if n <= 1:
        return {}
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization dict."""
    if not factors:
        return [1]
    primes = list(factors.keys())
    exponents = list(factors.values())
    divs = [1]
    for i, p in enumerate(primes):
        current = list(divs)
        mul = 1
        for _ in range(exponents[i]):
            mul *= p
            for d in current:
                divs.append(d * mul)
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        
        # 1/x + 1/y + 1/z = 4/n with x < y < z
        # Fix x, let d = 4x - n > 0
        # (d*y - nx)(d*z - nx) = (nx)^2
        # a = d*y - nx, b = d*z - nx, a*b = (nx)^2
        # y = (a + nx)/d, z = (b + nx)/d
        # Need: a % d == (-nx) % d, y > x, z > y, all distinct
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            d = 4 * x - n
            if d <= 0:
                continue
            
            nx = n * x
            N = nx * nx
            
            # Factorize N = (nx)^2
            factors_nx = _factorize(nx)
            factors_N = {}
            for p, e in factors_nx.items():
                factors_N[p] = e * 2
            
            divs = _divisors_from_factors(factors_N)
            
            target_mod = (-nx) % d
            
            for a in divs:
                if a >= nx:
                    continue
                if a % d != target_mod:
                    continue
                
                y_num = a + nx
                if y_num % d != 0:
                    continue
                y = y_num // d
                if y <= x:
                    continue
                
                b = N // a
                z_num = b + nx
                if z_num % d != 0:
                    continue
                z = z_num // d
                if z <= y:
                    continue
                if z > 10**12:
                    continue
                
                if x == y or y == z or x == z:
                    continue
                
                lhs = n * y * z + n * x * z + n * x * y
                rhs = 4 * x * y * z
                if lhs != rhs:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
