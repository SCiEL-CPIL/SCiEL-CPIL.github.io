import time
import math

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    
    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n
    
    def divisors(num):
        """Generate all divisors of num."""
        small = []
        large = []
        i = 1
        while i * i <= num:
            if num % i == 0:
                small.append(i)
                if i != num // i:
                    large.append(num // i)
            i += 1
        return small + large[::-1]
    
    def solve_for_n(n):
        """Find three distinct positive integers x,y,z with 1/x+1/y+1/z = 4/n."""
        # x must satisfy 4x > n, i.e., x > n/4
        x_min = n // 4 + 1
        # Upper bound on x: for very large x, 1/x is tiny, so 1/y+1/z ≈ 4/n
        # y and z would need to be around n/4 each, but they must be > x
        # Let's cap x reasonably; if x is too large, y,z must be close to n/4
        # but they need to be > x, so x < n/4 * something
        # Actually if x >= n, then 1/x <= 1/n, so 1/y+1/z >= 3/n, 
        # meaning y,z < 3n/2 roughly. But y > x >= n, so y,z in (n, 3n/2)
        # Let's just try x from x_min up to some reasonable limit
        # For x very large, a = 4x-n is large, b = nx is large
        # The key constraint: y > x, z > y (we can assume wlog x<y<z)
        # 1/y + 1/z = a/b where a=4x-n, b=nx
        # y+z = a*t, yz = b*t for some positive integer t
        # y,z > 0 requires a*t > 0 (a>0 ok) and b*t > 0 (ok)
        # y,z real requires D = a^2*t^2 - 4*b*t >= 0, i.e., a^2*t >= 4b
        # Also y,z integers requires D perfect square and a*t ± sqrt(D) even
        
        # Let's iterate x from x_min upward
        # For each x, a = 4*x - n, b = n*x
        # We need t such that:
        #   1. a^2 * t >= 4 * b  (D >= 0)
        #   2. D = a^2*t^2 - 4*b*t is a perfect square
        #   3. a*t + sqrt(D) is even (so y is integer)
        #   4. y = (a*t + sqrt(D))//2, z = (a*t - sqrt(D))//2
        #   5. y > x, z > y, all distinct, all <= 10^12
        
        # Alternative: iterate over possible values of (y-z) or use the divisor approach
        # From 1/y + 1/z = a/b:
        # b(y+z) = a*y*z
        # Let g = gcd(a,b), a' = a/g, b' = b/g
        # b'(y+z) = a'*y*z
        # This means a' | b'(y+z) and since gcd(a',b')=1, a' | (y+z)
        # Let y+z = a'*k for some k
        # Then b'*a'*k = a'*y*z => b'*k = y*z
        # So y,z are roots of u^2 - a'*k*u + b'*k = 0
        # D = a'^2*k^2 - 4*b'*k must be perfect square
        # y = (a'*k + sqrt(D))/2, z = (a'*k - sqrt(D))/2
        # Need y > x, z > y, all distinct
        
        # k must be such that D >= 0: a'^2*k^2 >= 4*b'*k => a'^2*k >= 4*b'
        # So k >= ceil(4*b'/a'^2)
        
        # Also for y,z to be integers: a'*k + sqrt(D) must be even
        # And y > x means (a'*k + sqrt(D))/2 > x
        
        # Let's iterate k starting from the minimum
        
        for x in range(x_min, x_min + 50000):  # limit x range for speed
            if time.time() > deadline:
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
            
            # Also need y > x: (a'*k + sqrt(D))/2 > x
            # This is automatically satisfied for large k, but let's check
            # For small k, y might be <= x
            
            # Iterate k from k_min upward, but limit iterations
            # D = a'^2*k^2 - 4*b'*k = k*(a'^2*k - 4*b')
            # For D to be perfect square, let's just check
            # Limit k iterations to keep time bounded
            max_k_iters = 500 if x < x_min + 1000 else 200
            for ki in range(k_min, k_min + max_k_iters):
                if time.time() > deadline:
                    return None
                k = ki
                D = a_prime * a_prime * k * k - 4 * b_prime * k
                if D < 0:
                    continue
                if not is_perfect_square(D):
                    continue
                sqrt_D = math.isqrt(D)
                # Check parity: a'*k + sqrt_D must be even
                num = a_prime * k + sqrt_D
                if num % 2 != 0:
                    continue
                y = num // 2
                z = (a_prime * k - sqrt_D) // 2
                if z <= 0:
                    continue
                # Check distinctness and ordering
                if x == y or y == z or x == z:
                    continue
                # We want all positive and <= 10^12
                if x > 10**12 or y > 10**12 or z > 10**12:
                    continue
                # Verify the equation
                if x * y * z != 4 * n * y * z // 1:  # This is wrong, let me redo
                    pass
                # 1/x + 1/y + 1/z = 4/n
                # Cross multiply: yz + xz + xy = 4n*x*y*z / (xyz) ... no
                # 1/x + 1/y + 1/z = (yz + xz + xy)/(xyz) = 4/n
                # So n*(yz + xz + xy) = 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    return [x, y, z]
        
        # Fallback: iterate y and solve for z
        # 1/y + 1/z = a/b => z = b*y / (a*y - b)
        # So a*y - b must divide b*y
        # Let d = a*y - b, then d | b*y
        # z = b*y/d, need z > y, integer, distinct from x,y
        
        # This might be faster for some cases
        # Let's try this as a fallback with limited y range
        y_start = x + 1
        y_end = x + 10000  # limit
        for y in range(y_start, min(y_end, x + 30000)):
            if time.time() > deadline:
                return None
            denom = a * y - b
            if denom <= 0:
                continue
            if (b * y) % denom != 0:
                continue
            z = (b * y) // denom
            if z <= y:
                continue
            if z > 10**12:
                continue
            if x == y or y == z or x == z:
                continue
            lhs = n * (y * z + x * z + x * y)
            rhs = 4 * x * y * z
            if lhs == rhs:
                return [x, y, z]
        
        return None
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        result = solve_for_n(n)
        answers[idx] = result
    
    # For unsolved n values, try a second pass with extended search
    remaining = [i for i in range(254) if answers[i] is None]
    for idx in remaining:
        if time.time() > deadline:
            break
        n = idx + 3
        result = solve_for_n(n)
        answers[idx] = result
    
    return answers
