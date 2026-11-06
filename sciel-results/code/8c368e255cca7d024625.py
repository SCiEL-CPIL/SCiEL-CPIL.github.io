import time
from functools import lru_cache

def factorize(n):
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

@lru_cache(maxsize=None)
def generate_divisors_from_factors_cached(factors_tuple):
    factors = dict(factors_tuple)
    divs = [1]
    for p, e in factors.items():
        current_divs = []
        for d in divs:
            val = d
            for _ in range(e + 1):
                current_divs.append(val)
                val *= p
        divs = current_divs
    return tuple(sorted(divs))

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
            
        found = None
        x_min = n // 4 + 1
        # Expand search range to find more solutions
        x_max = 20 * n + 100
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            
            # Check if A divides n*x in a useful way
            # For 1/y + 1/z = A/(n*x), we need n*x = y*z - y - z
            # So (y-1)(z-1) = n*x + 1
            # Thus we need divisors of n*x + 1
            B = n * x
            B_sq = B * B
            
            # Factorize B
            fac_B = factorize(B)
            # Factors of B^2 have exponents 2*e_i
            fac_B_sq = tuple((p, 2*e) for p, e in sorted(fac_B.items()))
            
            # Get cached divisors of B^2
            divisors = generate_divisors_from_factors_cached(fac_B_sq)
            
            for u in divisors:
                if u > B:
                    break  # Divisors are sorted, so we can break
                if (u + B) % A != 0:
                    continue
                
                v = B_sq // u
                if (v + B) % A != 0:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                # Check distinctness and bounds
                if x == y or x == z or y == z:
                    continue
                if y > 10**12 or z > 10**12:
                    continue
                if y < 1 or z < 1:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
