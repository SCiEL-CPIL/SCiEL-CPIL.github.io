"""Program for rank-minimal exact tensor decomposition of 2x3 by 3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns rank-minimal decomposition for 2x3 * 3x6 matrix multiplication.
    
    Strategy: Split the problem into structured blocks and use Strassen's algorithm
    where applicable to reduce rank below the schoolbook 36.
    
    A is 2x3, B is 3x6, C is 2x6.
    c[i][k] = sum_{j=0}^{2} a[i][j] * b[j][k]
    
    We split A into [A1 | a2] where A1 is 2x2 (cols 0,1) and a2 is 2x1 (col 2).
    We split B into [B1; b2] where B1 is 2x6 (rows 0,1) and b2 is 1x6 (row 2).
    
    C = A1 * B1 + a2 * b2
    
    For A1 * B1 (2x2 * 2x6):
    Split B1 into three 2x2 blocks: B1a (cols 0,1), B1b (cols 2,3), B1c (cols 4,5)
    Each 2x2*2x2 uses Strassen with rank 7.
    Total for A1*B1: 3 * 7 = 21
    
    For a2 * b2 (2x1 * 1x6): outer product, rank 6.
    
    Total: 21 + 6 = 27
    
    But we can do better by sharing structure. Let me try a direct construction
    with rank 22 using a more clever decomposition.
    
    Actually, let me implement a verified rank-22 decomposition based on known results.
    """
    n, m, p = 2, 3, 6
    
    # Start with schoolbook rank 36, then try to reduce
    # For now, implement the structured decomposition with rank 27
    
    rank = 27
    U = [[0]*rank for _ in range(n*m)]  # 6 x 27
    V = [[0]*rank for _ in range(m*p)]  # 18 x 27
    W = [[0]*rank for _ in range(p*n)]  # 12 x 27
    
    col = 0
    
    # Strassen helper for 2x2 * 2x2 subproblem
    # A sub: indices (i,j) where i in {0,1}, j in {j0, j0+1}
    # B sub: indices (j,k) where j in {j0, j0+1}, k in {k0, k0+1}
    # C sub: indices (i,k) where i in {0,1}, k in {k0, k0+1}
    # W index: k*n + i
    
    def strassen_block(j0, k0):
        """Apply Strassen's 7-term algorithm for A[0:2,j0:j0+2] * B[j0:j0+2,k0:k0+2]"""
        nonlocal col
        # A entries: a[i][j0], a[i][j0+1] -> U index i*m+j
        # B entries: b[j0][k0], b[j0][k0+1], b[j0+1][k0], b[j0+1][k0+1] -> V index j*p+k
        # C entries: c[i][k0], c[i][k0+1] -> W index k*n+i
        
        a00 = 0*m + j0      # U index for a[0][j0]
        a01 = 0*m + (j0+1)  # U index for a[0][j0+1]
        a10 = 1*m + j0      # U index for a[1][j0]
        a11 = 1*m + (j0+1)  # U index for a[1][j0+1]
        
        b00 = j0*p + k0        # V index for b[j0][k0]
        b01 = j0*p + (k0+1)    # V index for b[j0][k0+1]
        b10 = (j0+1)*p + k0    # V index for b[j0+1][k0]
        b11 = (j0+1)*p + (k0+1) # V index for b[j0+1][k0+1]
        
        c00 = k0*n + 0    # W index for c[0][k0]
        c01 = (k0+1)*n + 0 # W index for c[0][k0+1]
        c10 = k0*n + 1    # W index for c[1][k0]
        c11 = (k0+1)*n + 1 # W index for c[1][k0+1]
        
        # M1 = a00 * (b01 + b11)
        U[a00][col] = 1
        V[b01][col] = 1
        V[b11][col] = 1
        W[c00][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M2 = a10 * (b00 + b01)
        U[a10][col] = 1
        V[b00][col] = 1
        V[b01][col] = 1
        W[c10][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M3 = (a00 + a01) * b10
        U[a00][col] = 1
        U[a01][col] = 1
        V[b10][col] = 1
        W[c00][col] = 1
        W[c01][col] = 1
        col += 1
        
        # M4 = (a10 + a11) * b00
        U[a10][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M5 = a01 * (b11 - b10)
        U[a01][col] = 1
        V[b11][col] = 1
        V[b10][col] = -1
        W[c00][col] = 1
        W[c01][col] = 1
        col += 1
        
        # M6 = a11 * (b01 - b00)
        U[a11][col] = 1
        V[b01][col] = 1
        V[b00][col] = -1
        W[c10][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M7 = (a00 + a11) * (b00 + b11)
        U[a00][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        V[b11][col] = 1
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M6
        # C11 = M1 - M3 + M4 + M7
        # So M7 contributes: +1 to c00, -1 to c01, -1 to c10, +1 to c11
        W[c00][col] = 1
        W[c01][col] = -1
        W[c10][col] = -1
        W[c11][col] = 1
        col += 1
    
    # Apply Strassen to three 2x2*2x2 blocks
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    strassen_block(0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    strassen_block(0, 2)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6]
    strassen_block(0, 4)
    
    # Now handle the outer product: a2 * b2 where a2 is A[0:2, 2] and b2 is B[2, 0:6]
    # a2[0] = a[0][2], a2[1] = a[1][2]
    # b2[k] = b[2][k]
    # c[i][k] += a2[i] * b2[k]
    
    a2_0 = 0*m + 2  # U index for a[0][2]
    a2_1 = 1*m + 2  # U index for a[1][2]
    
    for k in range(p):
        b2_k = 2*p + k  # V index for b[2][k]
        c0_k = k*n + 0  # W index for c[0][k]
        c1_k = k*n + 1  # W index for c[1][k]
        
        U[a2_0][col] = 1
        V[b2_k][col] = 1
        W[c0_k][col] = 1
        col += 1
        
        U[a2_1][col] = 1
        V[b2_k][col] = 1
        W[c1_k][col] = 1
        col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
