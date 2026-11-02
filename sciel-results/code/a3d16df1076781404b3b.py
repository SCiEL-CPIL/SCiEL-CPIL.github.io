"""Matrix multiplication 2x2x10 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 40 (10 columns x 4 per 2x2x1 multiplication)
    rank = n * m * p  # 2*2*10 = 40
    
    U = [[0] * rank for _ in range(n * m)]   # shape (4, 40)
    V = [[0] * rank for _ in range(m * p)]   # shape (20, 40)
    W = [[0] * rank for _ in range(p * n)]   # shape (20, 40)
    
    # For each column k of B (0..p-1), we need 4 bilinear terms for the 2x2x1 multiplication
    # The 4 terms for multiplying A (2x2) by a column of B (2x1):
    #   C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #   C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]  (but here B is 2x1, so only 1 column per block)
    # 
    # Actually for 2x2 times 2x1:
    #   C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    #   C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # That's only 4 bilinear terms total, rank 4.
    #
    # But wait - the problem is 2x2 times 2x10.
    # U: (n*m, rank) = (4, rank), indices i*m+j where i=0..1, j=0..1
    # V: (m*p, rank) = (20, rank), indices j*p+k where j=0..1, k=0..9
    # W: (p*n, rank) = (20, rank), indices k*n+i where k=0..9, i=0..1
    #
    # C[k, i] = sum_j A[i, j] * B[j, k]
    # So C[k*n + i] = sum_j A[i*m+j] * B[j*p+k]
    #
    # Schoolbook: each (i,j,k) triple is one term
    # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1
    # Total: n*m*p = 40 terms
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
