"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 82.0  # safety margin before 90s kill
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z (distinct positive integers)
        # Assume x <= y <= z, then 3/x >= 4/n => x <= 3n/4
        # Also 1/x < 4/n => x > n/4
        # So x ranges from floor(n/4)+1 to floor(3n/4)
        
        x_min = n // 4 + 1
        x_max = 3 * n // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            if found is not None:
                break
            
            # 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
            # Let a = 4x - n, b = nx
            # Need a > 0 for positive y,z
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # 1/y + 1/z = a/b
            # (y+z)*a = b*y*z
            # y*z*a - y*b - z*b = 0
            # (ay - b)(az - b) = b^2
            # Let u = ay - b, v = az - b, then u*v = b^2
            # y = (u + b)/a, z = (v + b)/a
            # Need u and v to be divisors of b^2 such that:
            #   (u + b) % a == 0
            #   (v + b) % a == 0
            #   y = (u+b)//a, z = (v+b)//a
            #   y > x, z > y (to ensure distinct and ordered)
            #   u < v (so y < z)
            #   u*v = b^2
            #   u > 0, v > 0
            
            # Since u*v = b^2, we iterate over divisors u of b^2 with u <= b (since u <= v and u*v=b^2, u <= b)
            # Actually u can be up to b since if u > b then v < b and we'd have u > v, so we restrict u <= b.
            
            # But b can be large. Let's think about the range of y.
            # 1/y < a/b => y > b/a
            # 2/y >= a/b => y <= 2b/a
            # So y ranges from floor(b/a)+1 to floor(2b/a)
            
            y_min = b // a + 1
            y_max = 2 * b // a
            
            # Also need y > x
            y_min = max(y_min, x + 1)
            
            # Iterate over y in range
            # For each y, check if z = b*y / (a*y - b) is an integer > y
            # denominator = a*y - b must divide b*y
            # Let d = a*y - b
            # z = b*y / d
            # Need d | b*y and d > 0 (which is true since y > b/a)
            
            # To speed up, note that d = a*y - b, so y = (d + b)/a
            # d must be such that (d+b) % a == 0, i.e., d ≡ -b (mod a)
            # And d | b*y = b*(d+b)/a, so a*d | b*(d+b)
            # This is equivalent to: a*d divides b*d + b^2
            # i.e., a*d divides b^2 (since a*d divides b*d)
            # So d must divide b^2/gcd(a*d, b^2)... this is getting complex.
            
            # Simpler: just iterate y and check divisibility.
            # The range of y is at most 2b/a - b/a = b/a, which for small x is about n/4, manageable.
            
            for y in range(y_min, y_max + 1):
                d = a * y - b
                if d <= 0:
                    continue
                # Check if d divides b*y
                if (b * y) % d != 0:
                    continue
                z = (b * y) // d
                if z > y and z <= 10**12:
                    # Verify distinctness: x, y, z all distinct
                    if x != y and y != z and x != z:
                        found = [x, y, z]
                        break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
