import time

def construct():
    deadline = time.time() + 55
    answers = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        found = None
        x_min = n // 4 + 1
        # Heuristic limit for x
        x_limit = n + 200
        
        for x in range(x_min, x_limit + 1):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            B = n * x
            
            # y_max = floor(2 * B / A)
            y_max = (2 * B) // A
            
            for y in range(x + 1, y_max + 1):
                # z = B*y / (A*y - B)
                den = A * y - B
                if den <= 0:
                    continue
                
                num = B * y
                
                if num % den == 0:
                    z = num // den
                    if z > y:
                        if z <= 10**12:
                            found = [x, y, z]
                            break
            if found is not None:
                break
                
        answers[idx] = found
        
    return answers
