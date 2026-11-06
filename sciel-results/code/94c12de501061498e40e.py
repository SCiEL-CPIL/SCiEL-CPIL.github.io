import time

DENOM_BOUND = 10**12

def construct():
    deadline = time.time() + 85.0
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        found = None
        n_start = time.time()
        
        # x must satisfy 1/x < 4/n => x > n/4
        x_min = n // 4 + 1
        
        # Try x values starting from x_min
        # Solutions often exist for small x
        # We will try up to 100 x values or until time runs out
        for x in range(x_min, x_min + 100):
            if time.time() > deadline:
                break
            
            # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
            # Let A = 4x - n, B = nx
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            # y must be > x
            # 1/y < A/B => y > B/A
            # 1/y > A/B - 1/z. Since z > y, 1/z < 1/y, so 1/y > A/(2B) => y < 2B/A
            # So y is in (B/A, 2B/A)
            y_min = B // A + 1
            y_max = (2 * B) // A
            
            # Also y must be > x
            if y_min <= x:
                y_min = x + 1
                
            if y_min > y_max:
                continue
                
            # Iterate y
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                    
                # z = B*y / (A*y - B)
                # Denominator D = A*y - B
                D = A * y - B
                if D <= 0:
                    continue
                    
                Num = B * y
                if Num % D != 0:
                    continue
                    
                z = Num // D
                
                # Check conditions
                if z <= y:
                    continue
                if z > DENOM_BOUND:
                    continue
                if y > DENOM_BOUND:
                    continue
                if x > DENOM_BOUND:
                    continue
                    
                # Distinctness
                if x != y and y != z and x != z:
                    found = [x, y, z]
                    break
                    
            if found:
                break
                
        results[idx] = found
        
    return results
