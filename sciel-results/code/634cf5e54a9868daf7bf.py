import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    n = 10
    
    # Construct L0: Cyclic Latin Square
    L0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Construct L1: Orthogonal to L0
    # Uses the structure of Z_2 x Z_5
    L1 = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        i1 = i // 5
        i2 = i % 5
        for j in range(n):
            j1 = j // 5
            j2 = j % 5
            
            # v1 part (mod 2)
            v1 = (i1 + j1) % 2
            
            # v2 part (mod 5)
            # Using multiplier 2 for j2 to ensure orthogonality with L0's multiplier 1
            v2 = (i2 + 2 * j2) % 5
            
            L1[i][j] = 5 * v1 + v2

    squares = [L0, L1]
    
    # Ensure types are correct (Python int)
    result = []
    for s in squares:
        result.append([[int(x) for x in row] for row in s])
        
    return result
