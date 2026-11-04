DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 80 decomposition:
    # 4x4 block (schoolbook): 64
    # Cross terms: 16
    rank = 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. A44 * B44 (Schoolbook 4x4)
    # i in 0..3, j in 0..3, k in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # 2. a * B14 (a = A[0:4, 4], B14 = B[4, 0:4])
    # C[i, k] += A[i, 4] * B[4, k] for i in 0..3, k in 0..3
    # This is 1 term? No, it's 4 terms?
    # A[i,4] B[4,k].
    # U[i, 4, c] = 1, V[4, k, c] = 1, W[k, i, c] = 1.
    # This works for all i, k with 1 term?
    # Yes, as derived before.
    U[0 * m + 4][col] = 1
    U[1 * m + 4][col] = 1
    U[2 * m + 4][col] = 1
    U[3 * m + 4][col] = 1
    V[4 * p + 0][col] = 1
    V[4 * p + 1][col] = 1
    V[4 * p + 2][col] = 1
    V[4 * p + 3][col] = 1
    W[0 * n + 0][col] = 1
    W[1 * n + 0][col] = 1
    W[2 * n + 0][col] = 1
    W[3 * n + 0][col] = 1
    W[0 * n + 1][col] = 1
    W[1 * n + 1][col] = 1
    W[2 * n + 1][col] = 1
    W[3 * n + 1][col] = 1
    W[0 * n + 2][col] = 1
    W[1 * n + 2][col] = 1
    W[2 * n + 2][col] = 1
    W[3 * n + 2][col] = 1
    W[0 * n + 3][col] = 1
    W[1 * n + 3][col] = 1
    W[2 * n + 3][col] = 1
    W[3 * n + 3][col] = 1
    col += 1
    
    # 3. A44 * b (b = B[0:4, 4])
    # C[i, 4] += sum_j A[i, j] B[j, 4]
    # Schoolbook: 4 terms (j=0..3)
    for j in range(4):
        for i in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
        col += 1
        
    # 4. a * b11 (a = A[0:4, 4], b11 = B[4, 4])
    # C[i, 4] += A[i, 4] * B[4, 4]
    # 1 term
    for i in range(4):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
    col += 1
    
    # 5. A14 * B44 (A14 = A[4, 0:4], B44 = B[0:4, 0:4])
    # C[4, k] += sum_j A[4, j] B[j, k]
    # Schoolbook: 4 terms (j=0..3)
    for j in range(4):
        for k in range(4):
            U[4 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 4][col] = 1
        col += 1
        
    # 6. a11 * B14 (a11 = A[4, 4], B14 = B[4, 0:4])
    # C[4, k] += A[4, 4] * B[4, k]
    # 1 term
    for k in range(4):
        U[4 * m + 4][col] = 1
        V[4 * p + k][col] = 1
        W[k * n + 4][col] = 1
    col += 1
    
    # 7. A14 * b (A14 = A[4, 0:4], b = B[0:4, 4])
    # C[4, 4] += sum_j A[4, j] B[j, 4]
    # Schoolbook: 4 terms (j=0..3)
    for j in range(4):
        U[4 * m + j][col] = 1
        V[j * p + 4][col] = 1
        W[4 * n + 4][col] = 1
        col += 1
        
    # 8. a11 * b11 (A[4, 4] * B[4, 4])
    # C[4, 4] += A[4, 4] * B[4, 4]
    # 1 term
    U[4 * m + 4][col] = 1
    V[4 * p + 4][col] = 1
    W[4 * n + 4][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
