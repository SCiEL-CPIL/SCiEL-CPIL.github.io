import time

def construct():
    answers = [None] * 254
    LIMIT = 10**12
    start_time = time.time()
    DEADLINE = 55.0 # Leave 5s margin
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() - start_time > DEADLINE:
            break
            
        found = None
        # x range: n/4 < x < 3n/4
        # x must be integer, so x >= floor(n/4) + 1
        start_x = n // 4 + 1
        # x < 3n/4, so x <= ceil(3n/4) - 1 or just 3*n//4
        # Let's use 3*n//4 as upper bound inclusive, check condition inside or rely on y bound
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            # Check if 1/x < 4/n
            if 4 * x <= n:
                continue
                
            # y < 2nx / (4x - n)
            # Since y is integer, y <= floor( (2nx - 1) / (4x - n) )
            # But 2nx / (4x-n) is not integer usually.
            # Let D = 4x - n
            # y < 2nx / D
            # y_max = int(2nx / D)
            # We need strict inequality y < 2nx/D.
            # If 2nx is divisible by D, y_max = 2nx/D - 1.
            # Else y_max = floor(2nx/D).
            D = 4 * x - n
            if D <= 0:
                continue
            
            num_y = 2 * n * x
            if num_y % D == 0:
                y_max = num_y // D - 1
            else:
                y_max = num_y // D
                
            if y_max <= x:
                continue
                
            for y in range(x + 1, y_max + 1):
                # z = nxy / (4xy - nx - ny)
                num_z = n * x * y
                den_z = 4 * x * y - n * x - n * y
                
                if den_z <= 0:
                    continue
                    
                if num_z % den_z != 0:
                    continue
                    
                z = num_z // den_z
                
                if z <= y:
                    continue
                    
                if z > LIMIT:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found

    return answers
