"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def _factorize(n):
    """Return prime factorization of n as dict {prime: exponent}."""
    factors = {}
    if n <= 1:
        return factors
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _all_divisors(factors):
    """Generate all divisors from prime factorization dict."""
    divs = [1]
    for p, e in factors.items():
        powers = [p ** i for i in range(e + 1)]
        divs = [d * pw for d in divs for pw in powers]
    return divs

def construct():
    start_time = time.time()
    DEADLINE = 80.0
    BOUND = 10**12
    
    answers = []
    
    for n in range(3, 257):
        if time.time() - start_time > DEADLINE:
            answers.append(None)
            continue
        
        found = None
        
        x_min = n // 4 + 1
        x_max = 3 * n // 4 + 3
        
        # Phase 1: Try divisor-based search for multiple x values
        for x in range(x_min, min(x_max + 1, x_min + 100)):
            if time.time() - start_time > DEADLINE:
                break
            
            a = 4 * x - n
            b = n * x
            
            if a <= 0:
                continue
            
            b_sq = b * b
            bf = _factorize(b)
            b_sq_factors = {p: 2 * e for p, e in bf.items()}
            divs = _all_divisors(b_sq_factors)
            
            for d in divs:
                if time.time() - start_time > DEADLINE:
                    break
                
                num_y = d + b
                if num_y % a != 0:
                    continue
                y = num_y // a
                
                b2d = b_sq // d
                num_z = b2d + b
                if num_z % a != 0:
                    continue
                z = num_z // a
                
                if y <= 0 or z <= 0:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                if x == y or x == z or y == z:
                    continue
                if not (x < y < z):
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        # Phase 2: Fallback - direct y search for a few x values
        if found is None:
            for x in range(x_min, min(x_max + 1, x_min + 20)):
                if time.time() - start_time > DEADLINE:
                    break
                
                a = 4 * x - n
                b = n * x
                if a <= 0:
                    continue
                
                y_low = max(x + 1, b // a + 1)
                y_high = min(2 * b // a - 1, BOUND)
                
                if y_low > y_high:
                    continue
                
                max_iter = 20000
                for y in range(y_low, min(y_high + 1, y_low + max_iter)):
                    if time.time() - start_time > DEADLINE:
                        break
                    
                    denom = a * y - b
                    if denom <= 0:
                        continue
                    
                    if (b * y) % denom == 0:
                        z = (b * y) // denom
                        if z > y and z <= BOUND and x != y and y != z:
                            found = [x, y, z]
                            break
                
                if found:
                    break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
