"""Solve 1/x + 1/y + 1/z = 4/n for n=3..256 with distinct positive denominators."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    answers = [None] * 254  # index i corresponds to n = i+3
    
    for i in range(254):
        if time.time() > deadline:
            break
        n = i + 3
        found = _solve_n(n, deadline)
        if found:
            answers[i] = found
    
    return answers

def _solve_n(n, deadline):
    """Find distinct x,y,z with 1/x+1/y+1/z=4/n, x<y<z, all <= 10^12."""
    # We need 1/x + 1/y + 1/z = 4/n
    # Fix x, then 1/y + 1/z = (4x-n)/(nx) = a/b where a=4x-n, b=nx
    # Need a > 0, so x > n/4
    # 1/y + 1/z = a/b => b(y+z) = a*y*z => a*y*z - b*y - b*z = 0
    # => a*y*z - b*y - b*z + b^2/a = b^2/a (if a divides b^2)
    # => (a*y - b)(a*z - b) = b^2
    # So we need a | b^2, and then factor b^2/a into two factors (a*y-b)(a*z-b)
    
    # Try x values starting from smallest possible
    # x must satisfy: 1/x < 4/n (since y,z > x means 1/y, 1/z < 1/x)
    # Actually we need 3/x > 4/n => x < 3n/4 is NOT required; we need 1/x < 4/n => x > n/4
    # And we need x < y < z, so 1/x > 1/y > 1/z
    # Sum = 4/n, and 1/x is the largest term, so 3/x > 4/n => x < 3n/4
    # Also 1/x < 4/n => x > n/4
    
    # So x ranges from floor(n/4)+1 to floor(3n/4)-1 (approximately)
    # Let's be more careful: we need 4/n - 1/x to be positive and representable as 1/y+1/z with y>x, z>y
    
    x_min = n // 4 + 1
    x_max = min(3 * n // 4 - 1, 10**12)  # x must be < 3n/4 for 3/x > 4/n
    
    # Try a range of x values
    # For efficiency, try x values in order and limit the search
    for x in range(x_min, x_max + 1):
        if time.time() > deadline:
            return None
        
        a = 4 * x - n  # numerator of remaining sum
        b = n * x      # denominator of remaining sum
        
        if a <= 0:
            continue
        
        # Need a | b^2 for the factorization approach
        b2 = b * b
        if b2 % a != 0:
            continue
        
        c = b2 // a  # (a*y - b)(a*z - b) = c
        
        # Factor c into d1 * d2 where d1 = a*y - b, d2 = a*z - b
        # d1 > 0, d2 > 0, d1 <= d2 (since y <= z)
        # y = (d1 + b) / a must be integer, y > x
        # z = (d2 + b) / a must be integer, z > y
        
        # Enumerate divisors d1 of c where d1 <= sqrt(c)
        # d1 must satisfy: (d1 + b) % a == 0 and (d1 + b) // a > x
        # d2 = c // d1, and (d2 + b) % a == 0 and (d2 + b) // a > y
        
        # To find divisors efficiently, factor c or just iterate up to sqrt(c)
        # c can be very large (b^2/a where b=n*x, so b^2 ~ n^2*x^2)
        # For n up to 256, x up to ~192, b up to ~49152, b^2 ~ 2.4e9, c up to ~2.4e9
        # sqrt(c) up to ~49000, which is manageable
        
        # But we can be smarter: d1 = a*y - b, so d1 ≡ -b (mod a)
        # Let d1 = a*k - b for some k > 0
        # Then d1 * d2 = c, so d2 = c / d1
        # And d2 = a*z - b, so z = (d2 + b) / a
        # y = k
        
        # So we need: d1 = a*k - b divides c, and d2 = c/d1, and z = (d2+b)/a is integer > y=k
        
        # Iterate k starting from x+1 (since y > x)
        # d1 = a*k - b must be positive, so a*k > b => k > b/a = n*x/(4x-n)
        # For x > n/4, 4x-n > 0, so k > nx/(4x-n)
        
        # Let's just iterate over possible y values (k) and check
        # y must be > x, and 1/y + 1/z = a/b => z = b*y / (a*y - b)
        # Need a*y - b > 0 and (a*y - b) | b*y
        
        # y_min: a*y > b => y > b/a = n*x/(4*x-n)
        y_min = max(x + 1, (n * x) // (4 * x - n) + 1)
        # y_max: we need z > y, so 1/z < 1/y, so 1/y + 1/z < 2/y
        # a/b < 2/y => y < 2b/a = 2nx/(4x-n)
        y_max = min(2 * n * x // (4 * x - n) - 1, 10**12)
        
        if y_min > y_max:
            continue
        
        # Iterate y from y_min to y_max
        # Limit iterations to avoid long loops
        max_iters = 100000
        for y in range(y_min, min(y_max + 1, y_min + max_iters)):
            if time.time() > deadline:
                return None
            
            denom = a * y - b
            if denom <= 0:
                continue
            if (b * y) % denom != 0:
                continue
            z = (b * y) // denom
            if z > y and z <= 10**12:
                # Verify: 1/x + 1/y + 1/z == 4/n
                # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    return [x, y, z]
        
        # If we didn't find in the limited range, try a few more x values
        # but don't spend too much time
    
    return None

# EVOLVE-BLOCK-END
