"""Rewritten program with a stronger divisor-enumeration strategy."""
# EVOLVE-BLOCK-START
import math

def _factorize(num):
    """Return prime factorization of num as dict {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= num:
        while num % d == 0:
            factors[d] = factors.get(d, 0) + 1
            num //= d
        d += 1
    if num > 1:
        factors[num] = factors.get(num, 0) + 1
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divs = [1]
    for p, e in factors.items():
        current_divs = []
        pow_p = 1
        for i in range(e + 1):
            for d in divs:
                current_divs.append(d * pow_p)
            pow_p *= p
        divs = current_divs
    return divs

def construct():
    answers = []
    BOUND = 1000000000000  # 10^12
    
    for n in range(3, 257):
        found = None
        # Strategy: Fix x = n/4 + k for small k.
        # Solve 1/y + 1/z = (4x-n)/(nx) = a/b.
        # (ay-b)(az-b) = b^2.
        # Let u = ay-b, v = az-b. u*v = b^2.
        # u = -b (mod a), v = -b (mod a).
        # Iterate over divisors u of b^2.
        
        x_start = (n + 3) // 4  # ceil(n/4)
        # If n is divisible by 4, x_start = n/4. But a=0, invalid. So x > n/4.
        if n % 4 == 0:
            x_start = n // 4 + 1
        else:
            x_start = (n + 3) // 4
            
        # Try small x values
        for x in range(x_start, x_start + 100):
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # Factorize b to get divisors of b^2
            # b = n * x. n <= 256, x < 100 + 64 = 164. b < 42000.
            # b^2 < 1.7 * 10^9. Number of divisors is manageable.
            b_factors = _factorize(b)
            # Exponents for b^2 are 2 * exponents for b
            b2_factors = {p: 2*e for p, e in b_factors.items()}
            
            divs_b2 = _divisors_from_factors(b2_factors)
            
            for u in divs_b2:
                # Check u = -b (mod a)
                if (u + b) % a != 0:
                    continue
                
                y = (u + b) // a
                # Check y is integer (guaranteed by mod check) and positive
                if y <= x:
                    continue
                
                v = (b * b) // u
                # Check v = -b (mod a)
                if (v + b) % a != 0:
                    continue
                
                z = (v + b) // a
                if z <= y:
                    continue
                
                if z > BOUND:
                    continue
                
                # Check distinctness
                if x == y or y == z or x == z:
                    continue
                
                # Verify equation
                # 1/x + 1/y + 1/z = 4/n
                # n(yz + xz + xy) = 4xyz
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
