import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            D = 4 * x - n
            if D <= 0:
                continue
            
            P = n * x
            P_sq = P * P
            
            for U in range(1, P + 1):
                if P_sq % U != 0:
                    continue
                
                V = P_sq // U
                
                if (U + P) % D != 0:
                    continue
                if (V + P) % D != 0:
                    continue
                
                y = (U + P) // D
                z = (V + P) // D
                
                if x == y or x == z or y == z:
                    continue
                
                if y > 10**12 or z > 10**12:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
