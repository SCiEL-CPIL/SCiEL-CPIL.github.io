"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_TIME_LIMIT = 150  # seconds

def construct():
    global _start_time
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i (based on W[k*n+i, r])
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
