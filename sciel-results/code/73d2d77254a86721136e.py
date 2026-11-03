"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 68
    rank = 68
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Indices for A (3x3)
    # A[0,0] A[0,1] A[0,2]
    # A[1,0] A[1,1] A[1,2]
    # A[2,0] A[2,1] A[2,2]
    
    for block in range(4):
        # B columns: 2*block, 2*block+1
        c0 = 2 * block
        c1 = 2 * block + 1
        
        # B indices:
        # B[0, c0], B[0, c1]
        # B[1, c0], B[1, c1]
        # B[2, c0], B[2, c1]
        
        # C indices:
        # C[0, c0], C[0, c1]
        # C[1, c0], C[1, c1]
        # C[2, c0], C[2, c1]
        
        # 1. Strassen for A[0:2, 0:2] * B[0:2, c0:c1]
        # A22 = [[A00, A01], [A10, A11]]
        # B22 = [[B0c0, B0c1], [B1c0, B1c1]]
        
        # M1 = (A00 + A11) * (B0c0 + B1c1)
        U[0][col] = 1 # A00
        U[1][col] = 1 # A11 (Wait, A is flattened. A[1,1] is index 1*3+1=4? No, n=3, m=3. A[i,j] -> i*m+j. A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5, A[2,0]=6, A[2,1]=7, A[2,2]=8)
        
        # Correction on indices:
        # A[0,0]=0, A[0,1]=1, A[0,2]=2
        # A[1,0]=3, A[1,1]=4, A[1,2]=5
        # A[2,0]=6, A[2,1]=7, A[2,2]=8
        
        # B[0,c0]=0*8+c0, B[0,c1]=0*8+c1
        # B[1,c0]=1*8+c0, B[1,c1]=1*8+c1
        # B[2,c0]=2*8+c0, B[2,c1]=2*8+c1
        
        # C[k*n+i, r] -> C[i,k] is index k*3+i.
        # C[0,c0] -> c0*3+0
        # C[0,c1] -> c1*3+0
        # C[1,c0] -> c0*3+1
        # C[1,c1] -> c1*3+1
        # C[2,c0] -> c0*3+2
        # C[2,c1] -> c1*3+2
        
        # M1
        U[0][col] = 1
        U[4][col] = 1
        V[c0][col] = 1
        V[c1][col] = 1
        V[8+c0][col] = 1 # B[1,c0]
        V[8+c1][col] = 1 # B[1,c1]
        # Wait, M1 = (A00 + A11) * (B0c0 + B1c1)
        # B0c0 is B[0,c0]. B1c1 is B[1,c1].
        V[c0][col] = 1
        V[8+c1][col] = 1
        
        # Contribution to C:
        # X00 = M1 + M4 - M5 + M7
        # X01 = M3 + M5
        # X10 = M2 + M4
        # X11 = M1 - M2 + M3 + M6
        # C[0,c0] gets M1. C[1,c1] gets M1.
        W[c0*3+0][col] = 1
        W[c1*3+1][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B0c0
        U[3][col] = 1
        U[4][col] = 1
        V[c0][col] = 1
        W[c0*3+1][col] = 1 # X10
        W[c1*3+1][col] = -1 # X11 has -M2
        col += 1
        
        # M3 = A00 * B1c1
        U[0][col] = 1
        V[8+c1][col] = 1
        W[c1*3+0][col] = 1 # X01
        W[c1*3+1][col] = 1 # X11
        col += 1
        
        # M4 = A11 * (B1c0 - B0c0)
        U[4][col] = 1
        V[8+c0][col] = 1
        V[c0][col] = -1
        W[c0*3+0][col] = 1 # X00
        W[c0*3+1][col] = 1 # X10
        col += 1
        
        # M5 = (A00 + A01) * B1c0
        U[0][col] = 1
        U[1][col] = 1
        V[8+c0][col] = 1
        W[c0*3+0][col] = -1 # X00 has -M5
        W[c1*3+0][col] = 1 # X01 has M5
        col += 1
        
        # M6 = (A10 - A00) * (B0c0 + B0c1)
        U[3][col] = 1
        U[0][col] = -1
        V[c0][col] = 1
        V[c1][col] = 1
        W[c1*3+1][col] = 1 # X11 has M6
        col += 1
        
        # M7 = (A01 - A11) * (B1c0 + B1c1)
        U[1][col] = 1
        U[4][col] = -1
        V[8+c0][col] = 1
        V[8+c1][col] = 1
        W[c0*3+0][col] = 1 # X00 has M7
        col += 1
        
        # 2. Outer product a21 * b12
        # a21 = [A02, A12] -> indices 2, 5
        # b12 = [B2c0, B2c1] -> indices 16+c0, 16+c1
        
        # Y00 = A02 * B2c0
        U[2][col] = 1
        V[16+c0][col] = 1
        W[c0*3+0][col] = 1
        col += 1
        
        # Y01 = A02 * B2c1
        U[2][col] = 1
        V[16+c1][col] = 1
        W[c1*3+0][col] = 1
        col += 1
        
        # Y10 = A12 * B2c0
        U[5][col] = 1
        V[16+c0][col] = 1
        W[c0*3+1][col] = 1
        col += 1
        
        # Y11 = A12 * B2c1
        U[5][col] = 1
        V[16+c1][col] = 1
        W[c1*3+1][col] = 1
        col += 1
        
        # 3. Row product a12 * B22
        # a12 = [A20, A21] -> indices 6, 7
        # B22 = [[B0c0, B0c1], [B1c0, B1c1]]
        
        # Z0 = A20*B0c0 + A21*B1c0
        # Z1 = A20*B0c1 + A21*B1c1
        
        # N1 = A20 * B0c0
        U[6][col] = 1
        V[c0][col] = 1
        W[c0*3+2][col] = 1
        col += 1
        
        # N2 = A20 * B0c1
        U[6][col] = 1
        V[c1][col] = 1
        W[c1*3+2][col] = 1
        col += 1
        
        # N3 = A21 * B1c0
        U[7][col] = 1
        V[8+c0][col] = 1
        W[c0*3+2][col] = 1
        col += 1
        
        # N4 = A21 * B1c1
        U[7][col] = 1
        V[8+c1][col] = 1
        W[c1*3+2][col] = 1
        col += 1
        
        # 4. Scalar product a11 * b12
        # a11 = A22 -> index 8
        # b12 = [B2c0, B2c1] -> indices 16+c0, 16+c1
        
        # W0 = A22 * B2c0
        U[8][col] = 1
        V[16+c0][col] = 1
        W[c0*3+2][col] = 1
        col += 1
        
        # W1 = A22 * B2c1
        U[8][col] = 1
        V[16+c1][col] = 1
        W[c1*3+2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
