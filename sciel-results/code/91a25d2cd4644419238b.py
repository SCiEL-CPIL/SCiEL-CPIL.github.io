import time

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4 + 1
        
        # Try small x first
        x_upper1 = min(x_max, 10000)
        for x in range(x_min, x_upper1 + 1):
            a = 4 * x - n
            if a <= 0:
                continue
            
            b = n * x
            y_lo = b // a + 1
            y_hi = 2 * b // a - 1
            
            if y_lo <= x:
                y_lo = x + 1
            if y_hi < y_lo:
                continue
            
            # Cap y range
            if y_hi - y_lo > 200000:
                y_hi = y_lo + 200000
            
            for y in range(y_lo, y_hi + 1):
                denom = a * y - b
                if denom <= 0:
                    continue
                
                if (b * y) % denom == 0:
                    z = (b * y) // denom
                    if z > y and z <= BOUND:
                        if x != y and x != z and y != z:
                            found = [x, y, z]
                            break
            
            if found:
                break
            if time.time() > deadline:
                break
        
        # If not found, try larger x
        if not found and time.time() < deadline:
            x_start = max(x_max, x_upper1) + 1
            x_upper2 = 50000
            for x in range(x_start, x_upper2 + 1):
                a = 4 * x - n
                if a <= 0:
                    continue
                
                b = n * x
                y_lo = b // a + 1
                y_hi = 2 * b // a - 1
                
                if y_lo <= x:
                    y_lo = x + 1
                if y_hi < y_lo:
                    continue
                
                if y_hi - y_lo > 100000:
                    y_hi = y_lo + 100000
                
                for y in range(y_lo, y_hi + 1):
                    denom = a * y - b
                    if denom <= 0:
                        continue
                    
                    if (b * y) % denom == 0:
                        z = (b * y) // denom
                        if z > y and z <= BOUND:
                            if x != y and x != z and y != z:
                                found = [x, y, z]
                                break
                
                if found:
                    break
                if time.time() > deadline:
                    break
        
        answers[idx] = found
    
    return answers
