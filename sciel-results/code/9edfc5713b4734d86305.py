import time
import math

def construct():
    deadline = time.time() + 88.0
    answers = [None] * 254
    
    def solve_for_n(n, time_budget=0.3):
        """Find three distinct positive integers x,y,z with 1/x+1/y+1/z = 4/n.
        Returns (x,y,z) or None."""
        t0 = time.time()
        # x must satisfy 4x > n, i.e., x > n/4
        x_min = n // 4 + 1
        
        # Primary strategy: iterate x from x_min, and for each x,
        # iterate y from x+1 upward, computing z = b*y / (a*y - b)
        # where a = 4x - n, b = n*x
        # This is O(1) per (x,y) pair check.
        
        # For a given x, y must satisfy:
        # a*y > b  =>  y > b/a = n*x/(4x-n)
        # and z > y  =>  b*y/(a*y-b) > y  =>  b > a*y - b  =>  2b > a*y  =>  y < 2b/a
        # So y is in (b/a, 2b/a), i.e., y in (n*x/(4x-n), 2*n*x/(4x-n))
        
        for x in range(x_min, x_min + 200):
            if time.time() - t0 > time_budget or time.time() > deadline:
                return None
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # y must be > b/a for denom > 0
            y_lo = b // a + 1
            if y_lo <= x:
                y_lo = x + 1
            
            # y must be < 2b/a for z > y
            y_hi = (2 * b) // a
            if y_hi <= y_lo:
                continue
            
            # Limit y range for speed
            max_y_span = 50000
            y_hi = min(y_hi, y_lo + max_y_span)
            
            for y in range(y_lo, y_hi + 1):
                denom = a * y - b
                if denom <= 0:
                    continue
                num = b * y
                if num % denom != 0:
                    continue
                z = num // denom
                if z <= y:
                    continue
                if z > 10**12:
                    continue
                # Check distinctness
                if x == y or y == z or x == z:
                    continue
                # Verify: n*(yz + xz + xy) = 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    return [x, y, z]
        
        # Second strategy: use the k-based approach for remaining x values
        for x in range(x_min + 200, x_min + 500):
            if time.time() - t0 > time_budget or time.time() > deadline:
                return None
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            g = math.gcd(a, b)
            a_prime = a // g
            b_prime = b // g
            
            # k_min from D >= 0: a'^2 * k >= 4 * b'
            k_min = (4 * b_prime + a_prime * a_prime - 1) // (a_prime * a_prime)
            if k_min < 1:
                k_min = 1
            
            max_k_iters = 200
            for ki in range(k_min, k_min + max_k_iters):
                if time.time() - t0 > time_budget or time.time() > deadline:
                    return None
                k = ki
                D = a_prime * a_prime * k * k - 4 * b_prime * k
                if D < 0:
                    continue
                sqrt_D = math.isqrt(D)
                if sqrt_D * sqrt_D != D:
                    continue
                num = a_prime * k + sqrt_D
                if num % 2 != 0:
                    continue
                y = num // 2
                z = (a_prime * k - sqrt_D) // 2
                if z <= 0 or z <= y:
                    continue
                if x == y or y == z or x == z:
                    continue
                if x > 10**12 or y > 10**12 or z > 10**12:
                    continue
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    return [x, y, z]
        
        return None
    
    # First pass: quick solve for all n
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline - 10:
            break
        result = solve_for_n(n, time_budget=0.25)
        answers[idx] = result
    
    # Second pass: extended search for unsolved
    remaining = [i for i in range(254) if answers[i] is None]
    for idx in remaining:
        if time.time() > deadline - 5:
            break
        n = idx + 3
        result = solve_for_n(n, time_budget=0.4)
        answers[idx] = result
    
    # Third pass: even more extended for still-unsolved
    remaining = [i for i in range(254) if answers[i] is None]
    for idx in remaining:
        if time.time() > deadline - 2:
            break
        n = idx + 3
        # Custom deeper search
        x_min = n // 4 + 1
        t0 = time.time()
        for x in range(x_min, x_min + 1000):
            if time.time() - t0 > 2.0 or time.time() > deadline - 1:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            y_lo = b // a + 1
            if y_lo <= x:
                y_lo = x + 1
            y_hi = (2 * b) // a
            y_hi = min(y_hi, y_lo + 200000)
            for y in range(y_lo, y_hi + 1):
                if time.time() - t0 > 2.0 or time.time() > deadline - 1:
                    break
                denom = a * y - b
                if denom <= 0:
                    continue
                num = b * y
                if num % denom != 0:
                    continue
                z = num // denom
                if z <= y or z > 10**12:
                    continue
                if x == y or y == z or x == z:
                    continue
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    answers[idx] = [x, y, z]
                    break
            if answers[idx] is not None:
                break
    
    return answers
