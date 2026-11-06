import time
import math

def construct():
    deadline = time.time() + 75.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        
        x_min = n // 4 + 1
        x_max = min(3 * n // 4, BOUND)
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            b = n * x
            b2 = b * b
            
            # d must satisfy: d | b^2, d_min <= d < b, (d+b) % a == 0, (b^2/d + b) % a == 0
            # Since (d+b) % a == 0, d ≡ -b (mod a)
            # So d = -b + k*a for k = 1, 2, 3, ...
            # d >= d_min and d < b
            
            d_min = max(1, 4 * x * x - 2 * n * x + 1)
            if d_min < 1:
                d_min = 1
            
            # d = -b + k*a, so k*a >= d_min + b => k >= (d_min + b) / a
            # and d < b => -b + k*a < b => k*a < 2b => k < 2b/a
            k_min = (d_min + b + a - 1) // a  # ceil((d_min + b) / a)
            k_max = (2 * b - 1) // a  # floor((2b-1)/a), so k*a < 2b
            
            for k in range(k_min, k_max + 1):
                d = -b + k * a
                if d < d_min or d >= b:
                    continue
                # Check d | b^2
                if b2 % d != 0:
                    continue
                # (d+b) % a == 0 is guaranteed by construction
                y = (d + b) // a
                if y <= x:
                    continue
                bz = b2 // d
                if (bz + b) % a != 0:
                    continue
                z = (bz + b) // a
                if z <= y:
                    continue
                if z > BOUND:
                    continue
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
