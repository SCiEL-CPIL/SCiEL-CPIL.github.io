"""Direct approach for distinct three-term unit-fraction representations of 4/n."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 50.0  # safety margin before 60s/90s limit
    answers = []
    
    # We need to find x, y, z distinct positive integers <= 10^12
    # such that 1/x + 1/y + 1/z = 4/n for n in 3..256
    
    # Strategy: For each n, search over possible x values (smallest denominator)
    # For a given x, we need 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
    # Let a = 4x - n, b = nx. We need a > 0, so x > n/4, i.e., x >= floor(n/4)+1
    # 1/y + 1/z = a/b => (y+z)/(yz) = a/b => b(y+z) = a*yz
    # => a*yz - b*y - b*z = 0
    # => (ay - b)(az - b) = b^2
    # So we need to find factor pairs of b^2 such that both factors are > 0 and congruent to -b mod a... 
    # Actually: ay - b = d, az - b = b^2/d for divisor d of b^2
    # Then y = (d + b)/a, z = (b^2/d + b)/a
    # We need d + b ≡ 0 (mod a) and b^2/d + b ≡ 0 (mod a)
    # And y, z positive integers, distinct from each other and from x.
    
    # Simpler: iterate x from floor(n/4)+1 upward, and for each x iterate y from x+1 upward
    # and check if z is an integer.
    
    # To make this fast, note that for fixed x, y must be in a limited range.
    # 1/y + 1/z = a/b where a=4x-n, b=nx
    # Since y < z (wlog), 2/y > a/b => y < 2b/a
    # And y > x (for distinctness, assuming x is smallest)
    # Also y >= max(x+1, ceil(b/a) + 1) roughly
    
    for n in range(3, 257):
        found = None
        
        # Try multiple x values starting from the minimum
        x_min = n // 4 + 1
        
        # For each x, we can use the factorization approach to speed up
        # (ay - b)(az - b) = b^2
        # Let u = ay - b, v = az - b, then uv = b^2
        # y = (u+b)/a, z = (v+b)/a
        # We need u+b ≡ 0 (mod a) and v+b ≡ 0 (mod a), with u,v > 0
        
        for x in range(x_min, x_min + 500):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # We need u*v = b^2, u ≡ -b (mod a), v ≡ -b (mod a)
            # u = a*k - b for some k, so a*k - b > 0 => k > b/a
            # Also u < v wlog, so u <= b (since u*v = b^2)
            # So a*k - b <= b => a*k <= 2b => k <= 2b/a
            # Also u must divide b^2, and v = b^2/u must be integer
            # And v ≡ -b (mod a)
            
            # Alternative: just iterate y in the valid range
            y_lo = max(x + 1, b // a + 1)
            y_hi = min(2 * b // a, 10**12)
            if y_hi < y_lo:
                continue
            
            # For large ranges, use the factorization method
            # u = a*y - b, so as y goes from y_lo to y_hi, u goes from a*y_lo - b to a*y_hi - b
            # We need u | b^2
            # Let's just iterate y but with a tighter bound
            
            # Actually, for most small n, the first few x values will work quickly
            # Let's iterate y
            for y in range(y_lo, y_hi + 1):
                if time.time() > deadline:
                    break
                # 1/z = a/b - 1/y = (ay - b)/(by)
                num = a * y - b
                if num <= 0:
                    continue
                den = b * y
                if den % num != 0:
                    continue
                z = den // num
                if z <= y:
                    continue
                if z > 10**12:
                    continue
                # Check distinctness
                if x != y and y != z and x != z:
                    found = [x, y, z]
                    break
            if found:
                break
        
        if time.time() > deadline:
            # Fill remaining with None
            while len(answers) < 254:
                answers.append(None)
            return answers
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
