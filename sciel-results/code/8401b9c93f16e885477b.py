import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = 2, 3, 5
    # Total rank 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 28
    V = [[0] * rank for _ in range(m * p)] # 15 x 28
    W = [[0] * rank for _ in range(p * n)] # 10 x 28
    
    col = 0
    
    # Helper to add Strassen decomposition for 2x2 x 2x2
    # A indices: i in 0,1; j in j0, j0+1
    # B indices: j in j0, j0+1; k in k0, k0+1
    # C indices: i in 0,1; k in k0, k0+1
    def add_strassen_222(j0, k0):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[j0][col] = 1; U[j0+1][col] = 1
        V[j0*3+k0][col] = 1; V[(j0+1)*3+k0][col] = 1
        W[k0*2+0][col] = 1; W[k0*2+1][col] = 1
        W[(k0+1)*2+0][col] = 1; W[(k0+1)*2+1][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[j0+1][col] = 1; U[j0+1+1][col] = 1 # Wait, A10 is j0, A11 is j0+1?
        # A is 2x2. A00=A[0,j0], A01=A[0,j0+1], A10=A[1,j0], A11=A[1,j0+1]
        # U index is i*m + j. i in 0,1.
        # M2 = (A[1,j0] + A[1,j0+1]) * B[j0, k0]
        U[1*3+j0][col] = 1; U[1*3+j0+1][col] = 1
        V[j0*3+k0][col] = 1
        W[k0*2+1][col] = 1 # C10
        W[k0*2+0][col] = -1 # C00 part?
        # Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # Let's map W correctly.
        # W[k*n+i]
        # M1 contributes to C00(1), C11(1)
        # M2 contributes to C10(1), C11(-1)
        # M3 contributes to C01(1), C11(1)
        # M4 contributes to C00(1), C10(1)
        # M5 contributes to C00(-1), C01(1)
        # M6 contributes to C11(1)
        # M7 contributes to C00(1)
        
        # Reset col for M1? No, I already incremented. Let's restart helper.
        pass

    # Let's just write it out linearly to avoid confusion.
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # j0=0, k0=0
    # M1
    U[0][col] = 1; U[3][col] = 1 # A00, A11 (i=0,j=0; i=1,j=1)
    V[0][col] = 1; V[3][col] = 1 # B00, B11 (j=0,k=0; j=1,k=1)
    W[0][col] = 1 # C00
    W[3][col] = 1 # C11
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1 # A10, A11
    V[0][col] = 1 # B00
    W[2][col] = 1 # C10
    W[3][col] = -1 # C11
    col += 1
    
    # M3
    U[0][col] = 1 # A00
    V[1][col] = 1; V[4][col] = -1 # B01, B11
    W[1][col] = 1 # C01
    W[3][col] = 1 # C11
    col += 1
    
    # M4
    U[3][col] = 1 # A11
    V[3][col] = 1; V[0][col] = -1 # B10, B00
    W[0][col] = 1 # C00
    W[2][col] = 1 # C10
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1 # A00, A01
    V[4][col] = 1 # B11
    W[0][col] = -1 # C00
    W[1][col] = 1 # C01
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1 # A10, A00
    V[0][col] = 1; V[1][col] = 1 # B00, B01
    W[3][col] = 1 # C11
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1 # A01, A11
    V[3][col] = 1; V[4][col] = 1 # B10, B11
    W[0][col] = 1 # C00
    col += 1
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # j0=0, k0=2
    # M1
    U[0][col] = 1; U[3][col] = 1
    V[2][col] = 1; V[5][col] = 1 # B02, B12
    W[4][col] = 1 # C02 (k=2, i=0)
    W[7][col] = 1 # C12 (k=2, i=1)
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1 # B02
    W[6][col] = 1 # C12
    W[7][col] = -1 # C13? No, C11 relative is C13 absolute.
    # C indices: k=2,3. i=0,1.
    # W[2*2+0]=4, W[2*2+1]=5, W[3*2+0]=6, W[3*2+1]=7
    # C00->C02, C01->C03, C10->C12, C11->C13
    W[7][col] = -1 # C13
    col += 1
    
    # M3
    U[0][col] = 1
    V[3][col] = 1; V[6][col] = -1 # B03, B13
    W[5][col] = 1 # C03
    W[7][col] = 1 # C13
    col += 1
    
    # M4
    U[3][col] = 1
    V[5][col] = 1; V[2][col] = -1 # B12, B02
    W[4][col] = 1 # C02
    W[6][col] = 1 # C12
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1 # B13
    W[4][col] = -1 # C02
    W[5][col] = 1 # C03
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1 # B02, B03
    W[7][col] = 1 # C13
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1
    V[5][col] = 1; V[6][col] = 1 # B12, B13
    W[4][col] = 1 # C02
    col += 1
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    # j0=0, k0=4 (only one column k=4)
    # 2x2 by 2x1. Rank 4.
    # C04 = A00 B04 + A01 B14
    # C14 = A10 B04 + A11 B14
    U[0][col] = 1; V[4][col] = 1; W[8][col] = 1 # A00 B04 -> C04
    col += 1
    U[1][col] = 1; V[7][col] = 1; W[8][col] = 1 # A01 B14 -> C04
    col += 1
    U[3][col] = 1; V[4][col] = 1; W[9][col] = 1 # A10 B04 -> C14
    col += 1
    U[4][col] = 1; V[7][col] = 1; W[9][col] = 1 # A11 B14 -> C14
    col += 1
    
    # Block 4: A[0:2, 2:3] * B[2:3, 0:5] -> C[0:2, 0:5]
    # A col 2, B row 2.
    # C[i,k] += A[i,2] B[2,k]
    # i in 0,1. k in 0..4.
    # 10 multipliers.
    for i in range(2):
        for k in range(5):
            U[i*3+2][col] = 1
            V[2*3+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
