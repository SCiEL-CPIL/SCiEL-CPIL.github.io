"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook tensor decomposition for matrix multiplication.
    For n x m by m * p, the rank is n*m*p.
    U, V, W are matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = 120.0
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline - 10:
                    # Safety margin: return current best (which is complete for schoolbook)
                    # In a search context, this would return the best found so far.
                    # Here, the loop is fast enough to finish, but we guard against timeout.
                    pass 
                # U corresponds to A[i, j]
                U[i * m + j][col] = 1
                # V corresponds to B[j, k]
                V[j * p + k][col] = 1
                # W corresponds to C[i, k] -> index k*n + i
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
