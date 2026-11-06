"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    BOUND = 10**12
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        x_lo = n // 4 + 1
        x_hi = (3 * n + 3) // 4
        
        for x in range(x_lo, x_hi + 1):
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            # (ay-b)(az-b) = b^2
            # U = ay-b, V = az-b, UV = b^2
            # y = (U+b)/a, z = (V+b)/a
            # Need: U | b^2, (U+b) % a == 0, (V+b) % a == 0
            # y > x, z > y, all distinct, <= BOUND
            
            b_sq = b * b
            sqrt_b_sq = math.isqrt(b_sq)
            
            # U must be > 0, U < b (so that y < 2b/a, ensuring z > y)
            # Also y > x: (U+b)/a > x => U > ax - b = x(4x-n) - nx = x(4x-2n) = 2x(2x-n)
            min_U = 2 * x * (2 * x - n) + 1
            if min_U < 1:
                min_U = 1
            
            # U < b (to ensure V > U, hence z > y)
            max_U = b - 1
            
            if min_U > max_U:
                continue
            
            found_for_x = None
            
            # Iterate k such that U = k*a - b is in [min_U, max_U]
            # k*a - b >= min_U => k >= (min_U + b) / a
            # k*a - b <= max_U => k <= (max_U + b) / a
            k_lo = (min_U + b + a - 1) // a  # ceil((min_U+b)/a)
            k_hi = (max_U + b) // a           # floor((max_U+b)/a)
            
            if k_hi < k_lo:
                continue
            
            # Limit search to avoid timeout
            max_k_iter = 50000
            if k_hi - k_lo + 1 > max_k_iter:
                # If too many k values, sample or use alternative
                # Try direct y iteration for this x
                y_lo = max(x + 1, b // a + 1)
                y_hi = min(2 * b // a, BOUND)
                
                if y_lo <= y_hi:
                    # Try a subset of y values
                    step = max(1, (y_hi - y_lo + 1) // 10000)
                    for y in range(y_lo, y_hi + 1, step):
                        denom = a * y - b
                        if denom <= 0:
                            continue
                        num = b * y
                        if num % denom == 0:
                            z = num // denom
                            if z > y and z <= BOUND:
                                if x != y and y != z and x != z:
                                    found_for_x = [x, y, z]
                                    break
                    # If not found with step, try consecutive
                    if not found_for_x:
                        for y in range(y_lo, min(y_lo + 5000, y_hi + 1)):
                            denom = a * y - b
                            if denom <= 0:
                                continue
                            num = b * y
                            if num % denom == 0:
                                z = num // denom
                                if z > y and z <= BOUND:
                                    if x != y and y != z and x != z:
                                        found_for_x = [x, y, z]
                                        break
            
            if not found_for_x:
                for k in range(k_lo, k_hi + 1):
                    U = k * a - b
                    if U <= 0 or U > max_U:
                        continue
                    if b_sq % U != 0:
                        continue
                    V = b_sq // U
                    # Check divisibility
                    if (U + b) % a != 0:
                        continue
                    if (V + b) % a != 0:
                        continue
                    y = (U + b) // a
                    z = (V + b) // a
                    # Check constraints
                    if y <= x or z <= y:
                        continue
                    if y > BOUND or z > BOUND:
                        continue
                    if x == y or y == z or x == z:
                        continue
                    found_for_x = [x, y, z]
                    break
            
            if found_for_x:
                found = found_for_x
                break
        
        results[idx] = found
    
    return results
# EVOLVE-BLOCK-END
