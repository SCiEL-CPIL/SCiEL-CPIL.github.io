"""Direct approach for distinct three-term unit-fraction representations of 4/n."""
import time
import math
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 50.0  # safety margin before 60s/90s limit
    answers = []
    
    for n in range(3, 257):
        found = None
        x_min = n // 4 + 1
        
        # Try x values starting from minimum
        for x in range(x_min, x_min + 2000):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # We need (ay - b)(az - b) = b^2
            # Let u = ay - b, v = az - b, uv = b^2
            # u = ka - b for integer k, with u > 0 => k > b/a
            # u <= b (wlog u <= v) => ka - b <= b => k <= 2b/a
            # Also need u | b^2 and (b^2/u + b) % a == 0
            # And y = (u+b)/a > x, z = (v+b)/a > y, all distinct
            
            k_lo = b // a + 1
            k_hi = (2 * b) // a
            
            if k_hi < k_lo:
                continue
            
            # For each k in [k_lo, k_hi], check if u = ka - b divides b^2
            # and satisfies the congruence for v
            for k in range(k_lo, k_hi + 1):
                if time.time() > deadline:
                    break
                u = k * a - b
                if u <= 0:
                    continue
                # Check if u divides b^2
                if (b * b) % u != 0:
                    continue
                v = (b * b) // u
                # Check v + b ≡ 0 (mod a)
                if (v + b) % a != 0:
                    continue
                y = (u + b) // a
                z = (v + b) // a
                # Verify y, z are positive integers (they should be by construction)
                if y <= 0 or z <= 0:
                    continue
                # Check ordering: want x < y < z for distinctness
                if y <= x or z <= y:
                    continue
                # Check distinctness
                if x == y or y == z or x == z:
                    continue
                # Check bound
                if z > 10**12:
                    continue
                # Verify the equation
                # 1/x + 1/y + 1/z should equal 4/n
                # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            if found:
                break
        
        if time.time() > deadline:
            while len(answers) < 254:
                answers.append(None)
            return answers
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
