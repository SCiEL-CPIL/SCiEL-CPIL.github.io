"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # For M_{4,4,5}, use rank-20 decomposition:
    # Core M_{4,4,4}: rank 16 (one term per (i,k) pair)
    # Tail M_{4,4,1}: rank 4 (one term per i)
    # Total rank: 20
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]  # 16 x 20
    V = [[0] * rank for _ in range(m * p)]  # 20 x 20
    W = [[0] * rank for _ in range(p * n)]  # 20 x 20
    
    col = 0
    
    # Core: for each (i, k) in [0,3]x[0,3], add term
    # U[i*m+j, col] = 1 for all j
    # V[j*p+k, col] = 1 for all j  
    # W[k*n+i, col] = 1
    for i in range(n):
        for k in range(4):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Tail: for each i in [0,3], add term for column k=4
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # U[i*m+j, col] = 1 for all j
    # V[j*p+4, col] = 1 for all j
    # W[4*n+i, col] = 1
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
