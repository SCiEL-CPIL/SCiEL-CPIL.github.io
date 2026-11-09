import time
import math

def construct():
    answers = []
    deadline = time.time() + 55  # Safety margin before 60s limit
    BOUND = 10**12
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy n/4 < x < 3n/4 for distinct positive solutions with x < y < z
        x_start = n // 4 + 1
        x_end = (3 * n) // 4  # inclusive upper bound for x
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n  # numerator of (4/n - 1/x)
            b = n * x       # denominator of (4/n - 1/x)
            
            if a <= 0:
                continue
            
            # We need 1/y + 1/z = a/b with y > x, z > y
            # y = (d + b) / a must be integer, so (d + b) % a == 0
            # z = (b^2/d + b) / a must be integer
            
            # y_min from y > b/a and y > x
            y_min = max(b // a + 1, x + 1)
            # y_max from y < 2*b/a (ensures z > y)
            y_max = (2 * b - 1) // a  # strict inequality y < 2b/a
            
            if y_min > y_max:
                continue
            
            # Use step size to skip y values where (b*y) % (a*y - b) != 0
            # This is hard to exploit directly, so iterate but with early z bound check
            for y in range(y_min, y_max + 1):
                denom = a * y - b
                if denom <= 0:
                    continue
                # Quick check: z = b*y/denom, need z <= BOUND
                # b*y/denom <= BOUND => b*y <= BOUND * denom = BOUND*(a*y - b)
                # b*y <= BOUND*a*y - BOUND*b => BOUND*b <= y*(BOUND*a - b)
                # If BOUND*a - b <= 0, then no valid z (since BOUND*b > 0)
                if BOUND * a <= b:
                    break  # For larger y, z will only get smaller... wait, z = b*y/(a*y-b)
                # Actually z = b*y/(a*y-b). As y increases, a*y-b increases, so z decreases.
                # So if z > BOUND at current y, it might be <= BOUND for larger y.
                # But we need z > y, so y < 2b/a. Within this range, z is decreasing.
                
                if (b * y) % denom != 0:
                    continue
                z = (b * y) // denom
                if z <= y:
                    continue
                if z > BOUND:
                    continue
                # Check distinctness: x, y, z all distinct
                if x != y and y != z and x != z:
                    # Verify: 1/x + 1/y + 1/z == 4/n
                    # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
