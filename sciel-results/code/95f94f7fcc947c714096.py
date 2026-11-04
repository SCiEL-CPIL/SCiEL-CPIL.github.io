"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-16 exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Partition:
    - A = [a0; A1] where a0 is row 0 (1x3), A1 is rows 1-2 (2x3)
    - B = [B1 | B2] where B1 is cols 0-1 (3x2), B2 is cols 2-3 (3x2)
    
    C = [a0*B1, a0*B2; A1*B1, A1*B2]
    
    Each sub-product:
    - a0*B1: 1x2 output, rank 2
    - a0*B2: 1x2 output, rank 2
    - A1*B1: 2x2 output, rank 6
    - A1*B2: 2x2 output, rank 6
    Total rank: 16
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    rank = 16
    U = [[0]*rank for _ in range(n*m)]  # 9 x 16
    V = [[0]*rank for _ in range(m*p)]  # 12 x 16
    W = [[0]*rank for _ in range(p*n)]  # 12 x 16
    
    col = 0
    
    # a0*B1: C[0,k] for k in {0,1}, C[0,k] = sum_j A[0,j]*B[j,k]
    for j in range(3):
        # Term for C[0,0]
        U[0*m + j][col] = 1
        V[j*p + 0][col] = 1
        W[0*n + 0][col] = 1  # W[k*n + i] = W[0*3 + 0] = W[0]
        col += 1
        # Term for C[0,1]
        U[0*m + j][col] = 1
        V[j*p + 1][col] = 1
        W[1*n + 0][col] = 1  # W[1*3 + 0] = W[3]
        col += 1
    
    # a0*B2: C[0,k] for k in {2,3}, C[0,k] = sum_j A[0,j]*B[j,k]
    for j in range(3):
        # Term for C[0,2]
        U[0*m + j][col] = 1
        V[j*p + 2][col] = 1
        W[2*n + 0][col] = 1  # W[2*3 + 0] = W[6]
        col += 1
        # Term for C[0,3]
        U[0*m + j][col] = 1
        V[j*p + 3][col] = 1
        W[3*n + 0][col] = 1  # W[3*3 + 0] = W[9]
        col += 1
    
    # A1*B1: C[i,k] for i in {1,2}, k in {0,1}
    # C[i,k] = sum_j A[i,j]*B[j,k]
    for j in range(3):
        # Term for C[1,0]
        U[1*m + j][col] = 1
        V[j*p + 0][col] = 1
        W[0*n + 1][col] = 1  # W[0*3 + 1] = W[1]
        col += 1
        # Term for C[1,1]
        U[1*m + j][col] = 1
        V[j*p + 1][col] = 1
        W[1*n + 1][col] = 1  # W[1*3 + 1] = W[4]
        col += 1
        # Term for C[2,0]
        U[2*m + j][col] = 1
        V[j*p + 0][col] = 1
        W[0*n + 2][col] = 1  # W[0*3 + 2] = W[2]
        col += 1
        # Term for C[2,1]
        U[2*m + j][col] = 1
        V[j*p + 1][col] = 1
        W[1*n + 2][col] = 1  # W[1*3 + 2] = W[5]
        col += 1
    
    # A1*B2: C[i,k] for i in {1,2}, k in {2,3}
    # C[i,k] = sum_j A[i,j]*B[j,k]
    for j in range(3):
        # Term for C[1,2]
        U[1*m + j][col] = 1
        V[j*p + 2][col] = 1
        W[2*n + 1][col] = 1  # W[2*3 + 1] = W[7]
        col += 1
        # Term for C[1,3]
        U[1*m + j][col] = 1
        V[j*p + 3][col] = 1
        W[3*n + 1][col] = 1  # W[3*3 + 1] = W[10]
        col += 1
        # Term for C[2,2]
        U[2*m + j][col] = 1
        V[j*p + 2][col] = 1
        W[2*n + 2][col] = 1  # W[2*3 + 2] = W[8]
        col += 1
        # Term for C[2,3]
        U[2*m + j][col] = 1
        V[j*p + 3][col] = 1
        W[3*n + 2][col] = 1  # W[3*3 + 2] = W[11]
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
