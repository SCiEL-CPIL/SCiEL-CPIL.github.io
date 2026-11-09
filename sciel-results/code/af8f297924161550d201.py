"""Improved program using divisor enumeration for the two-term sum."""
# EVOLVE-BLOCK-START
import math

def _factorize(num):
    """Return list of (prime, exponent) for num."""
    factors = []
    d = 2
    while d * d <= num:
        if num % d == 0:
            exp = 0
            while num % d == 0:
                num //= d
                exp += 1
            factors.append((d, exp))
        d += 1 if d == 2 else 2 # Skip even numbers after 2
    if num > 1:
        factors.append((num, 1))
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from list of (prime, exponent)."""
    divs = [1]
    for p, e in factors:
        new_divs = []
        for d in divs:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    return divs

def construct():
    answers = []
    BOUND = 1000000000000  # 10^12
    
    for n in range(3, 257):
        found = None
        x_start = n // 4 + 1
        # Limit x search range; divisor method is fast, so we can try more x
        for x in range(x_start, x_start + 1000):
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            # Factor b to get factors of b^2
            fac_b = _factorize(b)
            fac_b2 = [(p, 2 * e) for p, e in fac_b]
            divs_b2 = _divisors_from_factors(fac_b2)
            
            # We need u * v = b^2 with u < v (to ensure y < z)
            # u ≡ -b (mod a)
            # y = (u + b) / a, z = (v + b) / a
            for u in divs_b2:
                # Since y < z implies u < v, and u*v=b^2, we must have u < b
                if u >= b:
                    continue
                
                # Check congruence
                if u % a != (-b) % a:
                    continue
                
                y_num = u + b
                if y_num % a != 0:
                    continue
                y = y_num // a
                
                if y <= x:
                    continue
                
                v = (b * b) // u
                z_num = v + b
                if z_num % a != 0:
                    continue
                z = z_num // a
                
                if z <= y:
                    continue
                
                if z > BOUND:
                    continue
                
                # Check distinctness
                if x == y or y == z or x == z:
                    continue
                
                # Verify equation: n(yz + xz + xy) = 4xyz
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
