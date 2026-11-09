import math
import time

def construct():
    deadline = time.time() + 80  # 80s safety margin for 90s limit
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        target = 4 / n
        
        # Try x values starting from a reasonable minimum
        # We need 1/x < 4/n, so x > n/4
        # Also x, y, z should be distinct and positive
        min_x = n // 4 + 1
        
        for x in range(min_x, min_x + 5000):  # Try up to 5000 values of x
            if time.time() > deadline:
                break
            if found:
                break
            
            a = 4 * x - n  # numerator of (4/n - 1/x)
            if a <= 0:
                continue
            b = n * x       # denominator
            
            # Reduce a/b
            g = math.gcd(a, b)
            a_red = a // g
            b_red = b // g
            
            # Now solve 1/y + 1/z = a_red / b_red
            # => a_red * y * z = b_red * (y + z)
            # => (a_red * y - b_red)(a_red * z - b_red) = b_red^2
            # Let u = a_red * y - b_red, v = a_red * z - b_red
            # u * v = b_red^2, and y = (u + b_red) / a_red, z = (v + b_red) / a_red
            # Need u, v positive divisors of b_red^2 with u <= v (WLOG)
            # and (u + b_red) % a_red == 0, (v + b_red) % a_red == 0
            
            b_sq = b_red * b_red
            
            # Find divisors of b_sq
            # For efficiency, iterate up to sqrt(b_sq)
            sqrt_b_sq = int(math.isqrt(b_sq))
            
            # We want y > x and z > y (to ensure distinctness and ordering)
            # y = (u + b_red) / a_red > x => u + b_red > a_red * x => u > a_red * x - b_red
            # Since a = a_red * g, b = b_red * g, a_red * x - b_red = (a*x - b)/g = (4x-n)*x/g - nx/g
            # This is getting complex; let's just check distinctness at the end
            
            found_for_x = False
            for u in range(1, sqrt_b_sq + 1):
                if time.time() > deadline:
                    break
                if b_sq % u != 0:
                    continue
                v = b_sq // u
                if u > v:
                    continue
                
                # y = (u + b_red) / a_red
                if (u + b_red) % a_red != 0:
                    continue
                y = (u + b_red) // a_red
                
                # z = (v + b_red) / a_red
                if (v + b_red) % a_red != 0:
                    continue
                z = (v + b_red) // a_red
                
                if y <= 0 or z <= 0:
                    continue
                
                # Ensure distinctness and positivity
                if x == y or x == z or y == z:
                    continue
                if y > 10**12 or z > 10**12:
                    continue
                
                # Verify the solution
                # 1/x + 1/y + 1/z == 4/n
                # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    # Ensure all are positive and within bounds
                    if x > 0 and y > 0 and z > 0 and x <= 10**12 and y <= 10**12 and z <= 10**12:
                        found = [x, y, z]
                        found_for_x = True
                        break
            
            if found_for_x:
                break
        
        answers.append(found)
    
    return answers
