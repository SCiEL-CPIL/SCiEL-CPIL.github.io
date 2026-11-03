"""Tensor decomposition for 2x5x6 matrix multiplication using ladder/shifted technique."""

def construct():
    n, m, p = 2, 5, 6
    
    # Ladder decomposition: use sum and difference of input rows
    # For each j, define S_j = A[0,j] + A[1,j], D_j = A[0,j] - A[1,j]
    # Then A[0,j] = (S_j + D_j)/2, A[1,j] = (S_j - D_j)/2
    # 
    # C[0,k] = sum_j [(S_j + D_j)/2 * B[j,k]]
    # C[1,k] = sum_j [(S_j - D_j)/2 * B[j,k]]
    #
    # This uses 2 terms per (j,k): one for S_j*B[j,k], one for D_j*B[j,k]
    # Coefficients in U: for S terms, U[0*m+j]=1, U[1*m+j]=1
    #                    for D terms, U[0*m+j]=1, U[1*m+j]=-1
    # Coefficients in V: V[j*p+k]=1 for both
    # Coefficients in W: for S terms, W[k*n+0]=1/2, W[k*n+1]=1/2
    #                    for D terms, W[k*n+0]=1/2, W[k*n+1]=-1/2
    #
    # Rank = 2mp = 60. Same as schoolbook but different structure.
    # To get rank < 60, we need a more clever construction.
    
    rank = 2 * m * p  # 60
    
    U = [[0] * rank for _ in range(n * m)]  # 10 x 60
    V = [[0] * rank for _ in range(m * p)]  # 30 x 60
    W = [[0] * rank for _ in range(p * n)]  # 12 x 60
    
    col = 0
    for j in range(m):
        for k in range(p):
            # Term for S_j * B[j,k]: contributes to both C[0,k] and C[1,k] with +1/2
            U[0 * m + j][col] = 1
            U[1 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 0.5
            W[k * n + 1][col] = 0.5
            col += 1
            
            # Term for D_j * B[j,k]: contributes to C[0,k] with +1/2 and C[1,k] with -1/2
            U[0 * m + j][col] = 1
            U[1 * m + j][col] = -1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 0.5
            W[k * n + 1][col] = -0.5
            col += 1
    
    return {"U": U, "V": V, "W": W}
