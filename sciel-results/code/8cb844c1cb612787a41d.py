import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy: n/4 < x < 3n/4 (for distinct positive solutions with x<y<z)
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # y must be > x and ay - b > 0, so y > b/a
            y_start = max(x + 1, b // a + 1)
            
            # Upper bound: since z > y, we have 1/z < 1/y, so 4/n - 1/x - 1/y > 1/z > 1/y
            # This gives 4/n - 1/x > 2/y, so y > 2/(4/n - 1/x) = 2nx/(4x-n) = 2b/a
            # But we also need z <= 10^12, which gives a looser bound.
            # For efficiency, cap y search. If y gets too large, z approaches b/a from above.
            # Let's set a reasonable upper bound based on the problem constraints.
            y_end = min(2 * b // a + 1, 10**7)  # Cap to avoid infinite loop
            
            for y in range(y_start, y_end + 1):
                if time.time() > deadline:
                    break
                
                denom = a * y - b
                if denom <= 0:
                    continue
                
                # Check if denom divides b*y
                if (b * y) % denom == 0:
                    z = (b * y) // denom
                    if z > y and z <= 10**12:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
