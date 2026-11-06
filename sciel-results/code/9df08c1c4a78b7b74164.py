import time
from functools import lru_cache

@lru_cache(maxsize=None)
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
    return tuple(sorted(factors.items()))

def generate_divisors_from_factors(factors):
    divs = [1]
    for p, e in factors:
        current_divs = []
        for d in divs:
            val = d
            for _ in range(e + 1):
                current_divs.append(val)
                val *= p
        divs = current_divs
    return divs

def construct():
    deadline = time.time() + 75.0
    answers = [None] * 254
    
    # Precompute factorizations for all possible B = n*x values
    # B ranges from 3*1 to 256*1290 ≈ 330240
    max_B = 256 * (5 * 256 + 10)
    divisor_cache = {}
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
            
        found = None
        x_min = n // 4 + 1
        
        for x in range(x_min, 5 * n + 10):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            B_sq = B * B
            
            # Get or compute divisors of B^2
            if B not in divisor_cache:
                fac_B = factorize(B)
                fac_B_sq = tuple((p, 2*e) for p, e in fac_B)
                divisors = generate_divisors_from_factors(fac_B_sq)
                divisor_cache[B] = divisors
            else:
                divisors = divisor_cache[B]
            
            # Sort divisors and find u <= B
            for u in divisors:
                if u > B:
                    continue
                if (u + B) % A != 0:
                    continue
                
                v = B_sq // u
                if (v + B) % A != 0:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
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
