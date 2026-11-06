import time
import math

def construct():
    deadline = time.time() + 80.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        
        x_min = n // 4 + 1
        
        # Try x values starting from x_min
        max_x_tries = 2000
        for x in range(x_min, x_min + max_x_tries):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # y_low from condition 1: y > b/a
            y_low_1 = b // a + 1
            # y_low from condition 3: y > x
            y_low_3 = x + 1
            # y_low from condition 4: z <= BOUND
            denom = BOUND * a - b
            if denom <= 0:
                continue
            y_low_4 = (BOUND * b + denom - 1) // denom
            y_low = max(y_low_1, y_low_3, y_low_4)
            
            # y_high from condition 2: y < 2*b/a
            y_high = (2 * b + a - 1) // a - 1
            
            if y_high < y_low:
                continue
            
            # Try several y values starting from y_low to find one where z is integer
            max_y_tries = 30
            y_end = min(y_low + max_y_tries, y_high + 1)
            for y in range(y_low, y_end):
                num = a * y - b
                if num <= 0:
                    continue
                if (b * y) % num != 0:
                    continue
                z = (b * y) // num
                if z > y and y > x and z <= BOUND:
                    # Cross-multiply to verify
                    lhs = n * y * z + n * x * z + n * x * y
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
