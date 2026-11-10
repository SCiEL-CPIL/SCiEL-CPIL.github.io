import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    M = 10**12
    answers = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        found = None
        x_min = n // 4 + 1
        # x must be such that there exists y < z
        # 3/x > 4/n => x < 3n/4
        x_max = (3 * n) // 4 - 1 
        if x_max < x_min:
            x_max = x_min - 1 # just in case, though for n>=3, 3n/4 > n/4+1 usually
            
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            K = 4 * x - n
            if K <= 0:
                continue
                
            # y > nx/K
            y_low1 = (n * x) // K + 1
            
            # y < 2nx/K  (from z > y)
            y_high = (2 * n * x - 1) // K
            
            if y_low1 > y_high:
                continue
            
            # Constraint z <= M
            # z = nxy / (yK - nx) <= M
            # nxy <= M(yK - nx)
            # nxy <= MyK - Mnx
            # Mnx <= y(MK - nx)
            # y >= Mnx / (MK - nx)
            
            denom_z = M * K - n * x
            if denom_z <= 0:
                # If MK <= nx, then z is always > M for any y > nx/K?
                # Let's check limit y -> nx/K. z -> inf.
                # If MK <= nx, then for all y > nx/K, y(MK-nx) <= y(0) = 0 < Mnx.
                # So z > M always.
                continue
            
            y_low2 = (M * n * x + denom_z - 1) // denom_z
            
            y_start = max(y_low1, y_low2)
            
            if y_start > y_high:
                continue
            
            # Iterate y
            for y in range(y_start, y_high + 1):
                D = y * K - n * x
                # D must divide nxy
                if (n * x * y) % D == 0:
                    z = (n * x * y) // D
                    if z > y and z <= M:
                        # Distinctness: x < y < z ensures x!=y, y!=z, x!=z
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers[idx] = found
        
    return answers
