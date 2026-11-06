"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z (distinct positive integers)
        # 1/x + 1/y + 1/z = 4/n
        # => 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
        # Let A = 4x - n, B = nx. We need 1/y + 1/z = A/B
        # => B(y+z) = A*y*z
        # => A*y*z - B*y - B*z = 0
        # => (A*y - B)(A*z - B) = B^2
        # So we need factor pairs (d, B^2/d) of B^2 such that:
        #   A*y - B = d  => y = (d + B)/A
        #   A*z - B = B^2/d  => z = (B^2/d + B)/A
        # with y, z positive integers, y < z, and y > x, z > y.
        
        # x must satisfy: 4/n > 1/x (so x > n/4) and we need A = 4x - n > 0
        # Also 1/x < 4/n => x > n/4
        # And we need 1/x + 1/y + 1/z = 4/n with y > x, z > y, so 1/x < 4/n - 1/y - 1/z < 4/n
        # More precisely, since y > x and z > y > x: 1/x > 1/y > 1/z
        # So 3/x > 4/n => x < 3n/4
        # And 1/x < 4/n => x > n/4
        # So x ranges from floor(n/4)+1 to ceil(3n/4)-1, but let's be a bit more careful.
        # Actually: 1/x + 1/y + 1/z = 4/n, with x < y < z
        # 1/x < 4/n (since 1/y + 1/z > 0), so x > n/4
        # 3/x > 1/x + 1/y + 1/z = 4/n (since 1/x > 1/y > 1/z), so x < 3n/4
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4  # inclusive upper bound approximately
        
        # For each x in [x_min, x_max], try to find y, z
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need (A*y - B)(A*z - B) = B^2
            # d is a divisor of B^2, d < B (since y < z implies A*y - B < A*z - B)
            # y = (d + B) / A must be integer, y > x
            # z = (B^2 // d + B) / A must be integer, z > y
            
            # To be efficient, we can iterate over divisors d of B^2 where d < B
            # But B can be large. Let's think differently.
            # 
            # Alternative: iterate y from x+1 upward.
            # 1/y + 1/z = A/B => 1/z = A/B - 1/y = (A*y - B)/(B*y)
            # => z = B*y / (A*y - B)
            # We need A*y - B > 0, i.e., y > B/A = n*x/(4x-n)
            # And z must be integer and z > y.
            #
            # z = B*y / (A*y - B)
            # For z to be integer, (A*y - B) must divide B*y.
            # Let g = gcd(A*y - B, B). Then we need (A*y - B)/g to divide y... 
            # Actually simpler: just check if B*y % (A*y - B) == 0.
            
            # y must be > B/A and y > x
            y_min = max(x + 1, B // A + 1)
            
            # Upper bound on y: since z > y, we have 1/y + 1/z < 2/y, so A/B < 2/y => y < 2B/A
            # Also z = B*y/(A*y-B) > y => B*y > A*y^2 - B*y => 2B*y > A*y^2 => y < 2B/A
            y_max = 2 * B // A
            
            if y_max < y_min:
                continue
                
            # Iterate y
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                denom = A * y - B
                if denom <= 0:
                    continue
                if (B * y) % denom != 0:
                    continue
                z = (B * y) // denom
                if z > y:
                    found = [x, y, z]
                    break
            if found:
                break
        
        results[idx] = found
    
    return results
# EVOLVE-BLOCK-END
