"""Rewritten program with a stronger divisor-enumeration strategy."""
# EVOLVE-BLOCK-START
import math

def _factorize(num):
    """Return list of (prime, exponent) pairs for num."""
    factors = []
    d = 2
    while d * d <= num:
        if num % d == 0:
            count = 0
            while num % d == 0:
                num //= d
                count += 1
            factors.append((d, count))
        d += 1 if d == 2 else 2
    if num > 1:
        factors.append((num, 1))
    return factors

def _get_divisors(factors):
    """Generate all divisors from factorization."""
    divs = [1]
    for p, e in factors:
        new_divs = []
        for d in divs:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divs = new_divs
    return divs

def construct():
    answers = []
    BOUND = 1000000000000  # 10^12
    
    for n in range(3, 257):
        found = None
        # Strategy: Iterate x. For each x, solve for y,z using divisors of b^2.
        x_start = n // 4 + 1
        # Limit x search to keep within time
        for x in range(x_start, x_start + 500):
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # We need (ay-b)(az-b) = b^2
            # Let u = ay-b, v = az-b. u*v = b^2.
            # u ≡ -b (mod a)
            # y = (u+b)/a, z = (v+b)/a
            # Conditions: y, z integers, y > x, z > y, z <= BOUND.
            
            # To be efficient, we only need to check divisors u of b^2.
            # Since y < z, u < v, so u < b.
            # Also y > x => (u+b)/a > x => u > ax - b.
            
            # Factorize b^2? Or just b?
            # Divisors of b^2 can be generated from divisors of b?
            # If b = p1^e1 ... pk^ek, b^2 = p1^(2e1) ... pk^(2ek).
            # We can factorize b and then double the exponents.
            
            factors_b = _factorize(b)
            factors_b2 = [(p, 2*e) for p, e in factors_b]
            divs_b2 = _get_divisors(factors_b2)
            
            # Filter divisors u
            # u must be < b (to ensure u < v, so y < z)
            # u must be > ax - b (to ensure y > x)
            # u must satisfy u ≡ -b (mod a)
            
            lower_bound_u = a * x - b
            if lower_bound_u < 1:
                lower_bound_u = 1
                
            # Sort divisors to break early? Or just iterate.
            # Number of divisors is small.
            for u in divs_b2:
                if u >= b:
                    continue
                if u <= lower_bound_u:
                    continue
                if (u + b) % a != 0:
                    continue
                y = (u + b) // a
                # Check y > x (should be guaranteed by u > ax-b, but check just in case)
                if y <= x:
                    continue
                v = b * b // u
                if (v + b) % a != 0:
                    continue
                z = (v + b) // a
                if z <= y:
                    continue
                if z > BOUND:
                    continue
                # Distinctness check
                if x != y and y != z and x != z:
                    # Verify
                    if n * (y * z + x * z + x * y) == 4 * x * y * z:
                        found = [x, y, z]
                        break
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
