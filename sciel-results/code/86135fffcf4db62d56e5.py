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
            # Need: a | (U+b), a | (V+b), y > x, z > y, all distinct, <= BOUND
            
            b_sq = b * b
            
            min_U = 2 * x * (2 * x - n) + 1  # U must be > ax - b
            if min_U < 1:
                min_U = 1
            
            found_for_x = None
            
            k_lo = (min_U + b + a - 1) // a  # ceil((min_U+b)/a)
            k_hi = (2 * b - 1) // a  # floor((2b-1)/a), so k*a-b < b
            if k_hi < k_lo:
                continue
            
            for k in range(k_lo, k_hi + 1):
                U = k * a - b
                if U <= 0:
                    continue
                if b_sq % U != 0:
                    continue
                V = b_sq // U
                # Check y and z are integers
                if (U + b) % a != 0:
                    continue
                if (V + b) % a != 0:
                    continue
                y = (U + b) // a
                z = (V + b) // a
                # Check distinctness and bounds
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
