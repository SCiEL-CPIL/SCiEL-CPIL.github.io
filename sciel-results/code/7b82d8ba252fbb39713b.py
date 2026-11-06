import time
import math

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        # x must satisfy: n/4 < x < 3n/4 (for distinct positive y,z > x WLOG)
        # Actually just need 4x - n > 0, i.e., x > n/4
        # And we want x < y < z, so 3/x > 4/n => x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4  # exclusive-ish, let's use 3n/4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # (Ay - B)(Az - B) = B^2
            # Let d = Ay - B, then Az - B = B^2 / d
            # y = (d + B) / A, z = (B^2/d + B) / A
            # Need d | B^2, (d+B) % A == 0, (B^2//d + B) % A == 0
            # y, z positive integers, distinct from x and each other
            
            B2 = B * B
            
            # Find divisors of B2
            # For efficiency, iterate d up to sqrt(B2)
            limit = int(math.isqrt(B2))
            for d in range(1, limit + 1):
                if B2 % d != 0:
                    continue
                
                d1 = d
                d2 = B2 // d
                
                for dd in (d1, d2):
                    if (dd + B) % A != 0:
                        continue
                    y = (dd + B) // A
                    if y <= 0:
                        continue
                    rem = B2 // dd  # This is Az - B
                    if (rem + B) % A != 0:
                        continue
                    z = (rem + B) // A
                    if z <= 0:
                        continue
                    # Check distinctness
                    if x == y or x == z or y == z:
                        continue
                    # Check bound
                    if x > 10**12 or y > 10**12 or z > 10**12:
                        continue
                    # Found a solution
                    found = [x, y, z]
                    break
                if found:
                    break
            
            if found:
                break
        
        results[idx] = found
    
    return results
