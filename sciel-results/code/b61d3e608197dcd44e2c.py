"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            y_min = max(x + 1, b // a + 1)
            y_max = (2 * b) // a
            
            for y in range(y_min, y_max + 1):
                denom = a * y - b
                if denom <= 0:
                    continue
                if (b * y) % denom != 0:
                    continue
                z = (b * y) // denom
                if z <= y:
                    continue
                if z > 10**12:
                    continue
                found = [x, y, z]
                break
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
