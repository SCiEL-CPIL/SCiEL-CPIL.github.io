"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Term 0: A[0,0] * B[0,j] -> C[0,j]
    U[0 * m + 0][0] = 1
    V[0 * p + 0][0] = 1
    for j in range(p):
        W[j * n + 0][0] = 1
        
    # Term 1: A[0,1] * B[1,j] -> C[0,j]
    U[0 * m + 1][1] = 1
    V[1 * p + 0][1] = 1
    for j in range(p):
        W[j * n + 0][1] = 1
        
    # Term 2: A[1,0] * B[0,j] -> C[1,j]
    U[1 * m + 0][2] = 1
    V[0 * p + 0][2] = 1
    for j in range(p):
        W[j * n + 1][2] = 1
        
    # Term 3: A[1,1] * B[1,j] -> C[1,j]
    U[1 * m + 1][3] = 1
    V[1 * p + 0][3] = 1
    for j in range(p):
        W[j * n + 1][3] = 1
        
    # Term 4: A[0,2] * B[2,j] -> C[0,j]
    U[0 * m + 2][4] = 1
    V[2 * p + 0][4] = 1
    for j in range(p):
        W[j * n + 0][4] = 1
        
    # Term 5: A[1,2] * B[2,j] -> C[1,j]
    U[1 * m + 2][5] = 1
    V[2 * p + 0][5] = 1
    for j in range(p):
        W[j * n + 1][5] = 1
        
    # Term 6: A[2,0] * B[0,j] -> C[2,j]
    U[2 * m + 0][6] = 1
    V[0 * p + 0][6] = 1
    for j in range(p):
        W[j * n + 2][6] = 1
        
    # Term 7: A[2,1] * B[1,j] -> C[2,j]
    U[2 * m + 1][7] = 1
    V[1 * p + 0][7] = 1
    for j in range(p):
        W[j * n + 2][7] = 1
        
    # Term 8: A[2,2] * B[2,j] -> C[2,j]
    U[2 * m + 2][8] = 1
    V[2 * p + 0][8] = 1
    for j in range(p):
        W[j * n + 2][8] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
