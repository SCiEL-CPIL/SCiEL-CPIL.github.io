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
            
            # y range: b/a < y < 2*b/a
            y_lo = b // a + 1
            y_hi = min(2 * b // a, BOUND)
            
            # Also y > x for distinctness
            if y_lo <= x:
                y_lo = x + 1
            
            if y_lo > y_hi:
                continue
            
            # For each y in range, compute z = b*y / (a*y - b)
            # Need (a*y - b) divides b*y
            found_for_x = None
            for y in range(y_lo, y_hi + 1):
                denom = a * y - b
                if denom <= 0:
                    continue
                num = b * y
                if num % denom == 0:
                    z = num // denom
                    if z > y and z <= BOUND:
                        # Check distinctness
                        if x != y and y != z and x != z:
                            found_for_x = [x, y, z]
                            break
            
            if found_for_x:
                found = found_for_x
                break
        
        results[idx] = found
    
    return results
# EVOLVE-BLOCK-END
