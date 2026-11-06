import math
import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    BOUND = 10**12
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        found = None
        x_lo = n // 4 + 1
        x_hi = (3 * n) // 4
        
        for x in range(x_lo, x_hi + 1):
            if time.time() > deadline:
                break
            if found:
                break
            
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            # U = ay - b, V = az - b, UV = b^2
            # y = (U+b)/a > x  =>  U > ax - b = x(a-n) = x(4x-2n) = 2x(2x-n)
            # z = (V+b)/a > y  =>  V > U  =>  U < b (since UV=b^2 and U<V means U^2 < b^2)
            # Also U > 0
            # U = k*a - b for integer k
            # U > 0 => k*a > b => k > b/a
            # U < b => k*a - b < b => k*a < 2b => k < 2b/a
            # U > 2x(2x-n) => k*a - b > 2x(2x-n) => k > (b + 2x(2x-n))/a
            
            min_U = max(1, 2 * x * (2 * x - n) + 1)
            
            k_lo = (min_U + b + a - 1) // a  # ceil((min_U + b) / a)
            k_hi = (2 * b - 1) // a           # floor((2b-1)/a) so k*a-b < b
            
            if k_hi < k_lo:
                continue
            
            b_sq = b * b
            
            for k in range(k_lo, k_hi + 1):
                if time.time() > deadline:
                    break
                U = k * a - b
                if U <= 0 or U >= b:
                    continue
                if b_sq % U != 0:
                    continue
                V = b_sq // U
                # Check integrality of y and z
                if (U + b) % a != 0:
                    continue
                if (V + b) % a != 0:
                    continue
                y = (U + b) // a
                z = (V + b) // a
                # Check ordering and bounds
                if y <= x or z <= y:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                # All distinct since x < y < z
                found = [x, y, z]
                break
        
        results[idx] = found
    
    return results
