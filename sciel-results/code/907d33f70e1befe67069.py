import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0 # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(4):
        if time.time() > deadline:
            break
        # ... construction logic ...
        
    return {"U": U, "V": V, "W": W}
