import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    answers = [None] * 254
    unsolved = []
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline - 5.0:
            break
        found = None
        
        # Primary search: try x values in increasing order
        x_start = max(1, n // 4 + 1)
        x_end = min(3 * n + 200, 5000)
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline - 2.0:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            y_min = max(x + 1, B // A + 1)
            y_max = min(2 * B // A - 1, y_min + 500000)
            
            if y_max <= y_min:
                continue
                
            for y in range(y_min, y_max + 1):
                denom = A * y - B
                if denom <= 0:
                    continue
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    if z > y and z != x and z <= 10**12:
                        found = [x, y, z]
                        break
            if found:
                break
        
        # Secondary search for larger x
        if not found and time.time() < deadline - 1.0:
            x_start2 = x_end + 1
            x_end2 = 10000
            for x in range(x_start2, x_end2 + 1):
                if time.time() > deadline - 0.5:
                    break
                A = 4 * x - n
                if A <= 0:
                    continue
                B = n * x
                
                y_min = max(x + 1, B // A + 1)
                y_max = min(2 * B // A - 1, y_min + 100000)
                
                if y_max <= y_min:
                    continue
                    
                for y in range(y_min, y_max + 1):
                    denom = A * y - B
                    if denom <= 0:
                        continue
                    if (B * y) % denom == 0:
                        z = (B * y) // denom
                        if z > y and z != x and z <= 10**12:
                            found = [x, y, z]
                            break
                if found:
                    break
        
        answers[idx] = found
        if not found:
            unsolved.append(idx)
    
    # Tertiary pass for unsolved cases
    if unsolved and time.time() < deadline - 0.5:
        for idx in unsolved:
            if time.time() > deadline - 0.1:
                break
            n = idx + 3
            found = None
            
            for x in range(max(1, n // 4 + 1), 20000):
                if time.time() > deadline - 0.1:
                    break
                A = 4 * x - n
                if A <= 0:
                    continue
                B = n * x
                
                y_min = max(x + 1, B // A + 1)
                y_max = min(2 * B // A - 1, y_min + 50000)
                
                if y_max <= y_min:
                    continue
                    
                for y in range(y_min, y_max + 1):
                    denom = A * y - B
                    if denom <= 0:
                        continue
                    if (B * y) % denom == 0:
                        z = (B * y) // denom
                        if z > y and z != x and z <= 10**12:
                            found = [x, y, z]
                            break
                if found:
                    break
            
            if found:
                answers[idx] = found
    
    return answers
