"""Solve 1/x + 1/y + 1/z = 4/n for n=3..256 with distinct positive denominators."""
import time
import math

def _factorize(num):
    """Return list of (prime, exponent) for num."""
    factors = []
    d = 2
    while d * d <= num:
        if num % d == 0:
            cnt = 0
            while num % d == 0:
                num //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1
    if num > 1:
        factors.append((num, 1))
    return factors

def _gen_divisors(factors):
    """Generate all divisors from prime factorization."""
    divs = [1]
    for p, e in factors:
        new_divs = []
        pe = 1
        for i in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        
        x_min = n // 4 + 1
        x_max = (3 * n - 1) // 4
        
        if x_min > x_max:
            answers.append(None)
            continue
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            if found:
                break
            
            a = 4 * x - n
            b = n * x
            
            if a <= 0:
                continue
            
            # (ay-b)(az-b) = b^2
            # Let u = ay-b, v = az-b, so u*v = b^2
            # u < v, u > 4x^2 - 2nx (for y > x)
            # u ≡ -b (mod a)
            lower_bound = 4 * x * x - 2 * n * x
            b2 = b * b
            
            # Factorize b^2 and enumerate divisors
            # b = n * x, so factorize b then square exponents
            b_factors = _factorize(b)
            b2_factors = [(p, 2*e) for p, e in b_factors]
            
            divs = _gen_divisors(b2_factors)
            
            for u in divs:
                if u <= lower_bound:
                    continue
                if u >= b:
                    continue
                if b2 % u != 0:
                    continue
                v = b2 // u
                if v <= u:
                    continue
                if (v + b) % a != 0:
                    continue
                y = (u + b) // a
                z = (v + b) // a
                if x == y or y == z or x == z:
                    continue
                if y <= x or z <= y:
                    continue
                if y > 10**12 or z > 10**12:
                    continue
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
