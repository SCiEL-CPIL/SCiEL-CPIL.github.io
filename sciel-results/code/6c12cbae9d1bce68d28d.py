"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 120
    
    n, m, p = 4, 5, 6
    U_rows = n * m  # 20
    V_rows = m * p  # 30
    W_rows = p * n  # 24
    
    # Start with schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(U_rows)]
    V = [[0.0] * rank for _ in range(V_rows)]
    W = [[0.0] * rank for _ in range(W_rows)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank decomposition using a simple optimization approach
    # For now, return the schoolbook decomposition as it's guaranteed to work
    # A more advanced approach would use tensor rank minimization algorithms
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
