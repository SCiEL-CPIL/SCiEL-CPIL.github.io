"""Matrix multiplication 2x2 by 2x13 using Strassen blocks."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    # U shape (n*m, rank) = (4, rank)
    # V shape (m*p, rank) = (26, rank)
    # W shape (p*n, rank) = (26, rank)
    # Index conventions:
    #   U[i*m + j, c]  -> row i*m+j, col c
    #   V[j*p + k, c]  -> row j*p+k, col c
    #   W[k*n + i, c]  -> row k*n+i, col c
    # Matrix product C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] = 1 (product tensor)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # where W[k*n+i] corresponds to output C[i,k]
    
    # Use Strassen for 6 blocks of 2 columns (columns 0..11), schoolbook for column 12.
    # Strassen rank 7 per 2x2x2 block, schoolbook rank 4 for 2x2x1.
    # Total rank = 6*7 + 4 = 46
    
    rank = 46
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(26)]
    W = [[0]*rank for _ in range(26)]
    
    col = 0
    for block in range(6):
        k0 = 2 * block  # first column of this block
        k1 = k0 + 1     # second column of this block
        for s in range(7):
            c = col + s
            if s == 0:
                # M1 = (a00+a11)(b00+b11)
                U[0][c] = 1; U[3][c] = 1
                V[0][c] = 1; V[k1][c] = 1
                W[k0*2][c] = 1; W[k0*2+1][c] = 1
                W[k1*2][c] = -1; W[k1*2+1][c] = -1
            elif s == 1:
                # M2 = (a10+a11)b00
                U[2][c] = 1; U[3][c] = 1
                V[0][c] = 1
                W[k0*2+1][c] = 1
                W[k1*2+1][c] = -1
            elif s == 2:
                # M3 = a00(b01-b11)
                U[0][c] = 1
                V[k1][c] = 1
                W[k0*2+1][c] = 1
                W[k1*2+1][c] = 1
            elif s == 3:
                # M4 = a11(b00+b10)
                U[3][c] = 1
                V[0][c] = 1; V[k0+2][c] = 1
                W[k0*2][c] = -1
                W[k1*2+1][c] = 1
            elif s == 4:
                # M5 = (a00+a01)b11
                U[0][c] = 1; U[1][c] = 1
                V[k1][c] = 1
                W[k0*2][c] = 1
            elif s == 5:
                # M6 = (a10-a00)(b00+b01)
                U[2][c] = 1; U[0][c] = -1
                V[0][c] = 1; V[k0+2][c] = 1
                W[k1*2+1][c] = 1
            elif s == 6:
                # M7 = (a01-a11)(b10+b11)
                U[1][c] = 1; U[3][c] = -1
                V[k0+2][c] = 1; V[k1+2][c] = 1
                W[k0*2][c] = 1; W[k1*2+1][c] = 1
        col += 7
    
    # Schoolbook for last column (k=12)
    k = 12
    for j in range(2):
        for i in range(2):
            c = col
            U[i*2+j][c] = 1
            V[j*26+k][c] = 1
            W[k*2+i][c] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
