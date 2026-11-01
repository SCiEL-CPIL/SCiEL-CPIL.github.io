"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 3
    rank = 26
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Strassen terms for the top-left 2x2 block (indices 0,1 for i,j,k)
    # A indices: 00=0, 01=1, 10=3, 11=4
    # B indices: 00=0, 01=1, 10=3, 11=4
    # C indices: 00=0, 01=1, 10=3, 11=4 (W[k*3+i])
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1  # C00
    W[4][0] = 1  # C11
    
    # M2 = (A10 + A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1  # C10
    W[4][1] = -1 # C11
    
    # M3 = A00(B10 - B11)
    U[0][2] = 1
    V[3][2] = 1; V[4][2] = -1
    W[1][2] = 1  # C01
    W[4][2] = 1  # C11
    
    # M4 = A11(B01 - B00)
    U[4][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00
    W[3][3] = 1  # C10
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1 # C00
    W[1][4] = 1  # C01
    
    # M6 = (A10 - A00)(B00 + B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1  # C11
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1  # C00
    W[1][6] = 1  # C01
    
    # 2. Schoolbook terms for the remaining 19 positions
    # These are all (i, j, k) where not (i<2 and j<2 and k<2)
    col = 7
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Skip the 2x2 block handled by Strassen
                if i < 2 and j < 2 and k < 2:
                    continue
                
                # A[i, j] -> U[i*3+j]
                # B[j, k] -> V[j*3+k]
                # C[i, k] -> W[k*3+i]
                
                U[i * 3 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
