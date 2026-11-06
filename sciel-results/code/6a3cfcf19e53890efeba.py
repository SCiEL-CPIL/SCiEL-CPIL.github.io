import time
import math

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254  # indices 0..253 correspond to n=3..256
    
    for n in range(3, 257):
        if time.time() > deadline:
            break
        idx = n - 3
        found = None
        
        x_min = n // 4 + 1
        # x must satisfy 4*x > n, i.e., x >= n//4 + 1
        # Also, for 1/x + 1/y + 1/z = 4/n with x <= y <= z:
        # 3/x >= 4/n => x <= 3n/4
        x_max = (3 * n) // 4
        
        # We iterate x from x_min to x_max
        # For each x, we have 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
        # Let a = 4x - n, b = nx
        # Then 1/y + 1/z = a/b
        # => z = b*y / (a*y - b)
        # For z to be integer, a*y - b must divide b*y
        # Since gcd(a, b) may not be 1, let's use the standard approach.
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            b = n * x
            # We need 1/y + 1/z = a/b with y > x and z > y
            # y must be > b/a (since 1/y < a/b => y > b/a)
            # and y <= b/a * 2 (since 1/y > a/b / 2 => y < 2b/a)
            # More precisely:
            # 1/z = a/b - 1/y = (ay - b)/(by)
            # z = by/(ay - b)
            # For z > y: by/(ay-b) > y => b/(ay-b) > 1 => b > ay - b => 2b > ay => y < 2b/a
            # For z > 0: ay - b > 0 => y > b/a
            
            y_min = b // a + 1
            y_max = (2 * b) // a  # y < 2b/a, so y <= floor(2b/a - epsilon)
            # Actually y < 2b/a means y <= ceil(2b/a) - 1 if 2b/a is not integer, 
            # or 2b/a - 1 if it is integer. Let's use int(2*b/a) but be careful.
            # Since y must be integer and y < 2b/a, y_max = int(2*b/a) if 2*b/a not integer, 
            # else int(2*b/a) - 1.
            # A safe upper bound: y_max = int(2*b/a)
            # But we also need z > y, which is guaranteed by y < 2b/a.
            # And we need z integer.
            
            # To speed up, we can note that ay - b must divide b*y.
            # Let d = gcd(a, b). Then a' = a/d, b' = b/d.
            # 1/y + 1/z = a'/b'
            # z = b'*y / (a'*y - b')
            # For z integer, a'*y - b' must divide b'*y.
            # Since gcd(a', b') = 1, a'*y - b' must divide b'.
            # Let k = a'*y - b'. Then k | b'.
            # y = (k + b') / a'
            # So k must be a divisor of b' such that (k + b') is divisible by a'.
            
            d = math.gcd(a, b)
            a_prime = a // d
            b_prime = b // d
            
            # Find divisors of b_prime
            # b_prime can be large, but for small n it's manageable.
            # However, for n up to 256, b = n*x, x up to ~192, so b up to ~49152.
            # b_prime <= b. So divisors are manageable.
            
            # Instead of iterating y, iterate over divisors k of b_prime.
            # k must satisfy:
            # 1. k | b_prime
            # 2. (k + b_prime) % a_prime == 0
            # 3. y = (k + b_prime) // a_prime > x
            # 4. z = b_prime * y // k > y (which is equivalent to y < 2*b_prime/a_prime, 
            #    and since k = a'*y - b', this is automatically satisfied if y > b'/a')
            
            # Let's find all divisors of b_prime.
            # For efficiency, we can precompute divisors or just iterate up to sqrt(b_prime).
            
            # But iterating divisors for each x might be slow if done naively.
            # Let's try the y-iteration approach first with the divisibility trick.
            # Actually, iterating divisors of b_prime is likely faster for large ranges of y.
            
            # Let's collect divisors of b_prime.
            divisors = []
            for i in range(1, int(math.isqrt(b_prime)) + 1):
                if b_prime % i == 0:
                    divisors.append(i)
                    if i != b_prime // i:
                        divisors.append(b_prime // i)
            
            for k in divisors:
                if (k + b_prime) % a_prime != 0:
                    continue
                y = (k + b_prime) // a_prime
                if y <= x:
                    continue
                z = b_prime * y // k
                if z <= y:
                    continue
                # Check distinctness: x, y, z must be distinct
                if x == y or y == z or x == z:
                    continue
                # Check bound
                if x <= 10**12 and y <= 10**12 and z <= 10**12:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
