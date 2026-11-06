import time
import math

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_max, x_min - 1, -1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            y_lo = max(x + 1, B // A + 1)
            y_hi = (2 * B - 1) // A
            
            if y_hi < y_lo:
                continue
            
            for y in range(y_lo, y_hi + 1):
                denom = A * y - B
                if denom <= 0:
                    continue
                if (B * y) % denom != 0:
                    continue
                z = (B * y) // denom
                if z <= y:
                    continue
                if z > 10**12:
                    continue
                found = [x, y, z]
                break
            
            if found:
                break
        
        results[idx] = found
    
    return results
