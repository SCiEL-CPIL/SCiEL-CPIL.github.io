"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def construct():
    n, m, p = DIMENSIONS
    # A is 2x4, B is 4x5, C is 2x5
    # Schoolbook: 2*4*5 = 40 terms
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    elapsed = time.time() - _start_time
    if elapsed > 140:
        raise TimeoutError("Time limit exceeded")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
