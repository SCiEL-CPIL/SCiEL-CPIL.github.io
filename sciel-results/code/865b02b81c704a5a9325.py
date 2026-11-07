"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    A = []
    B = [0] * 61
    B[0] = 1
    n = 0
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
        new_B = [0] * (n + 2)
        for k in range(n + 1):
            if k == 0:
                new_B[0] = B[0]
            else:
                new_B[k] = B[k] | (B[k-1] << x)
        
        valid = True
        for i in range(n + 2):
            for j in range(i + 1, n + 2):
                if new_B[i] & new_B[j]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            B = new_B
            A.append(x)
            n += 1
    
    return list(A)
# EVOLVE-BLOCK-END
