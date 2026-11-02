"""Rank-17 exact tensor decomposition for 2x3x3 matrix multiplication."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(6)]
    
    col = 0
    # Strassen 7 products for 2x2 * 2x2 (A[:,0:2] * B[0:2,0:2])
    # Standard Strassen:
    # M1 = (a00+a11)*(b01+b12)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b11-b12)
    # M4 = a11*(b00-b10)
    # M5 = (a00+a01)*b12
    # M6 = (a10-a00)*(b00+b11)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1+M4-M5-M7, C01 = M3+M1+M7, C10 = M2+M4+M6, C11 = M1-M2+M3+M5
    
    # M1 = (a00+a11)*(b01+b12)
    U[0][col]=1; U[3][col]=1
    V[1][col]=1; V[4][col]=1
    W[0][col]=1; W[1][col]=1  # C00, C01
    col+=1
    
    # M2 = (a10+a11)*b00
    U[3][col]=1; U[4][col]=1
    V[0][col]=1
    W[2][col]=1; W[3][col]=1  # C10, C11
    col+=1
    
    # M3 = a00*(b11-b12)
    U[0][col]=1
    V[4][col]=1; V[5][col]=-1
    W[1][col]=1; W[0][col]=-1  # C01, C00 (negative)
    col+=1
    
    # M4 = a11*(b00-b10)
    U[3][col]=1
    V[0][col]=1; V[3][col]=-1
    W[2][col]=1; W[3][col]=1  # C10, C11
    col+=1
    
    # M5 = (a00+a01)*b12
    U[0][col]=1; U[1][col]=1
    V[5][col]=1
    W[0][col]=-1; W[2][col]=-1  # C00, C10 (negative)
    col+=1
    
    # M6 = (a10-a00)*(b00+b11)
    U[3][col]=1; U[0][col]=-1
    V[0][col]=1; V[4][col]=1
    W[2][col]=1; W[3][col]=1  # C10, C11
    col+=1
    
    # M7 = (a01-a11)*(b10+b11)
    U[1][col]=1; U[3][col]=-1
    V[3][col]=1; V[4][col]=1
    W[1][col]=1; W[3][col]=1  # C01, C11
    col+=1
    
    # Schoolbook for A[:,0:2] * B[0:2,2] (2x2 * 2x1)
    # a00*b02
    U[0][col]=1; V[2][col]=1; W[4][col]=1  # C02
    col+=1
    # a10*b02
    U[3][col]=1; V[2][col]=1; W[5][col]=1  # C12
    col+=1
    # a01*b12
    U[1][col]=1; V[5][col]=1; W[4][col]=1  # C02
    col+=1
    # a11*b12
    U[4][col]=1; V[5][col]=1; W[5][col]=1  # C12
    col+=1
    
    # Schoolbook for A[:,2] * B[2,:] (2x1 * 1x3)
    # a02*b20
    U[2][col]=1; V[6][col]=1; W[0][col]=1  # C00
    col+=1
    # a02*b21
    U[2][col]=1; V[7][col]=1; W[1][col]=1  # C01
    col+=1
    # a02*b22
    U[2][col]=1; V[8][col]=1; W[4][col]=1  # C02
    col+=1
    # a12*b20
    U[5][col]=1; V[6][col]=1; W[2][col]=1  # C10
    col+=1
    # a12*b21
    U[5][col]=1; V[7][col]=1; W[3][col]=1  # C11
    col+=1
    # a12*b22
    U[5][col]=1; V[8][col]=1; W[5][col]=1  # C12
    col+=1
    
    assert col == rank
    
    # Verify
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for _ in range(20):
        A_test = np.random.randn(2, 3)
        B_test = np.random.randn(3, 3)
        C_test = A_test @ B_test
        C_decomp = np.zeros((2, 3))
        for i in range(2):
            for k in range(3):
                s = 0.0
                for j in range(3):
                    for r in range(rank):
                        s += U_np[i*3+j, r] * V_np[j*3+k, r] * W_np[k*2+i, r]
                C_decomp[i, k] = s
        if not np.allclose(C_test, C_decomp, atol=1e-10):
            raise ValueError(f"Decomposition failed: max error = {np.max(np.abs(C_test - C_decomp))}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
