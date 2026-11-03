"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U: A[i, j] -> index i*m + j
                u_idx = i * m + j
                U[u_idx][col] = 1
                
                # V: B[j, k] -> index j*p + k
                v_idx = j * p + k
                V[v_idx][col] = 1
                
                # W: C[i, k] -> index k*n + i (trace convention)
                w_idx = k * n + i
                W[w_idx][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
