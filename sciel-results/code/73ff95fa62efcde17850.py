import time
import math
from math import gcd

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    BOUND = 10**12
    
    # Cache for divisors of q^2 (where q is the reduced denominator)
    divisor_sq_cache = {}
    
    def get_divisors_of_sq(q):
        """Return list of divisors of q^2."""
        if q in divisor_sq_cache:
            return divisor_sq_cache[q]
        n = q
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        
        divs = [1]
        for p, e in factors.items():
            exp = 2 * e
            new_divs = []
            for dv in divs:
                val = dv
                for _ in range(exp + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        divisor_sq_cache[q] = divs
        return divs
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        
        # x must satisfy: 1/x < 4/n (so x > n/4) and 1/x > 2/n (so x < n/2)
        # More precisely: x_min = floor(n/4) + 1, x_max = floor(n/2)
        # Actually: 1/x < 4/n => x > n/4, and we need 4/n - 1/x > 0 and < 2/n for y,z to exist
        # 4/n - 1/x < 2/n => 1/x > 2/n => x < n/2
        # So x in (n/4, n/2), i.e., x_min = n//4 + 1, x_max = (n-1)//2
        
        x_min = n // 4 + 1
        x_max = (n - 1) // 2
        
        # Try x values; prefer those where gcd(4x-n, nx) is large (simpler 2-term problem)
        # Build list of (x, gcd_val) and sort by gcd descending (larger gcd = easier)
        x_candidates = []
        for x in range(x_min, x_max + 1):
            a = 4 * x - n
            b = n * x
            g = gcd(a, b)
            x_candidates.append((g, x))
        
        # Sort: larger gcd first (easier 2-term problem), then smaller x
        x_candidates.sort(key=lambda t: (-t[0], t[1]))
        
        for g_val, x in x_candidates:
            if time.time() > deadline:
                break
            a = 4 * x - n
            b = n * x
            g = g_val
            # Reduced: 1/y + 1/z = (a/g) / (b/g)
            p = a // g
            q = b // g
            # y = q/t + p, z = q/t + q^2/t / p ... 
            # Actually: 1/y + 1/z = p/q => y(z+y) = qz => yz + y^2 - qz = 0
            # => y(y+z) = qz => let's use the standard: 
            # (y - q)(z - q) = q^2
            # So y = q + t, z = q + q^2/t for t | q^2
            # But we need y > x and z > y and distinct
            
            divs = get_divisors_of_sq(q)
            
            for t in divs:
                if t > q:
                    continue  # z would be < y or not valid ordering
                # y = q + t, z = q + q^2//t
                y = q + t
                z = q + q * q // t
                # Check: y > x, z > y, all distinct, z <= BOUND
                if y <= x:
                    continue
                if z <= y:
                    continue
                if z > BOUND:
                    continue
                # Verify: 1/x + 1/y + 1/z == 4/n via cross multiplication
                # x*y*z*4 == n*(y*z + x*z + x*y)
                lhs = 4 * x * y * z
                rhs = n * (y * z + x * z + x * y)
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
