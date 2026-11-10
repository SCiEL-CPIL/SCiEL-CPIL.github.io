"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if (x - x_min) % 50 == 0 and time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            
            # y = m, z = nxm/(ma-nx), need ma-nx | nxm, z > m, distinct
            m_max = (2 * n * x - 1) // a
            if m_max < 1:
                continue
            
            # Cap m_max to avoid excessive loops
            if m_max > 100000:
                m_max = 100000
            
            for m in range(1, m_max + 1):
                D = m * a - n * x
                if D <= 0:
                    continue
                num = n * x * m
                if num % D != 0:
                    continue
                z = num // D
                
                if z <= m:
                    continue
                if x == m or m == z or x == z:
                    continue
                if z > 10**12:
                    continue
                
                found = [x, m, z]
                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
