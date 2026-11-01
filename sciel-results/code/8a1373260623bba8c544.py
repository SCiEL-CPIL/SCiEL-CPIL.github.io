"""Matrix multiplication 2,2,11 with rank 39 (Strassen-based)."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-39 exact tensor decomposition for 2x2 * 2x11 matrix multiplication.
    
    Uses Strassen's rank-7 algorithm for five 2x2x2 blocks (covering columns 0-9 in pairs)
    and schoolbook rank-4 for the last column (column 10).
    
    Output factor W follows convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 11
    rank = 39  # 5 * 7 (Strassen) + 4 (schoolbook)
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 39
    V = [[0] * rank for _ in range(m * p)]  # 22 x 39
    W = [[0] * rank for _ in range(p * n)]  # 22 x 39
    
    # Strassen's algorithm for 2x2 blocks:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b11)
    # M4 = a22(b22-b12)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U indices: a11=0, a12=1, a21=2, a22=3
    # V indices for block at column offset k0: b11=k0, b12=k0+1, b21=p+k0, b22=p+k0+1
    # W indices: c11=k0*n+0, c12=(k0+1)*n+0, c21=k0*n+1, c22=(k0+1)*n+1
    
    # For each Strassen block (5 blocks covering column pairs 0-1, 2-3, 4-5, 6-7, 8-9)
    for block in range(5):
        k0 = 2 * block  # starting column of this block
        base = 7 * block  # starting rank column
        
        # M1 = (a11+a22)(b11+b22)
        U[0][base + 0] = 1  # a11
        U[3][base + 0] = 1  # a22
        V[k0][base + 0] = 1       # b11
        V[p + k0 + 1][base + 0] = 1  # b22
        W[k0 * n + 0][base + 0] += 1      # c11
        W[(k0 + 1) * n + 1][base + 0] += 1  # c22
        
        # M2 = (a21+a22)b11
        U[2][base + 1] = 1  # a21
        U[3][base + 1] = 1  # a22
        V[k0][base + 1] = 1  # b11
        W[k0 * n + 1][base + 1] += 1      # c21
        W[(k0 + 1) * n + 1][base + 1] -= 1  # c22
        
        # M3 = a11(b21-b11)
        U[0][base + 2] = 1  # a11
        V[p + k0][base + 2] = 1      # b21
        V[k0][base + 2] = -1         # b11
        W[(k0 + 1) * n + 0][base + 2] += 1  # c12
        W[(k0 + 1) * n + 1][base + 2] += 1  # c22
        
        # M4 = a22(b22-b12)
        U[3][base + 3] = 1  # a22
        V[p + k0 + 1][base + 3] = 1  # b22
        V[k0 + 1][base + 3] = -1     # b12
        W[k0 * n + 0][base + 3] += 1      # c11
        W[k0 * n + 1][base + 3] += 1      # c21
        
        # M5 = (a11+a12)b22
        U[0][base + 4] = 1  # a11
        U[1][base + 4] = 1  # a12
        V[p + k0 + 1][base + 4] = 1  # b22
        W[k0 * n + 0][base + 4] -= 1      # c11
        W[(k0 + 1) * n + 0][base + 4] += 1  # c12
        
        # M6 = (a21-a11)(b11+b12)
        U[2][base + 5] = 1   # a21
        U[0][base + 5] = -1  # a11
        V[k0][base + 5] = 1       # b11
        V[k0 + 1][base + 5] = 1   # b12
        W[(k0 + 1) * n + 1][base + 5] += 1  # c22
        
        # M7 = (a12-a22)(b21+b22)
        U[1][base + 6] = 1   # a12
        U[3][base + 6] = -1  # a22
        V[p + k0][base + 6] = 1       # b21
        V[p + k0 + 1][base + 6] = 1   # b22
        W[k0 * n + 0][base + 6] += 1  # c11
    
    # Schoolbook for last column (k=10), rank columns 35-38
    base = 35
    k0 = 10
    
    # c11 = a11*b11 + a12*b21  (for single column, c10 = a11*b10 + a12*b20)
    # Actually for 2x1, C[0] = A[0][0]*B[0][0] + A[0][1]*B[1][0]
    # C[1] = A[1][0]*B[0][0] + A[1][1]*B[1][0]
    
    # Term 1: a11 * b10 -> c10
    U[0][base + 0] = 1
    V[k0][base + 0] = 1
    W[k0 * n + 0][base + 0] = 1
    
    # Term 2: a12 * b20 -> c10
    U[1][base + 1] = 1
    V[p + k0][base + 1] = 1
    W[k0 * n + 0][base + 1] = 1
    
    # Term 3: a21 * b10 -> c20
    U[2][base + 2] = 1
    V[k0][base + 2] = 1
    W[k0 * n + 1][base + 2] = 1
    
    # Term 4: a22 * b20 -> c20
    U[3][base + 3] = 1
    V[p + k0][base + 3] = 1
    W[k0 * n + 1][base + 3] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
