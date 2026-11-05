import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Rank 28 decomposition
    # 1. A1 (2x2) * B11 (2x2) -> Strassen (7)
    # 2. A1 (2x2) * B12 (2x2) -> Strassen (7)
    # 3. A1 (2x2) * B13 (2x1) -> Schoolbook (4)
    # 4. A2 (2x1) * B2 (1x5) -> Schoolbook (10)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    # A indices: i in {0,1}, j in {j_off, j_off+1}
    # B indices: j in {j_off, j_off+1}, k in {k_off, k_off+1}
    # C indices: i in {0,1}, k in {k_off, k_off+1}
    def add_strassen(j_off, k_off):
        nonlocal col
        # M1
        U[0*3+0][col] += 1; U[1*3+1][col] += 1
        V[0*5+0][col] += 1; V[1*5+1][col] += 1
        W[0*2+0][col] += 1; W[1*2+1][col] += 1
        col += 1
        # M2
        U[1*3+0][col] += 1; U[1*3+1][col] += 1
        V[0*5+1][col] += 1
        W[0*2+1][col] += 1
        col += 1
        # M3
        U[0*3+0][col] += 1
        V[1*5+0][col] += 1; V[0*5+0][col] += -1
        W[1*2+0][col] += 1
        col += 1
        # M4
        U[0*3+0][col] += 1; U[1*3+0][col] += -1
        V[1*5+1][col] += 1; V[0*5+1][col] += 1
        W[0*2+0][col] += 1; W[0*2+1][col] += -1; W[1*2+1][col] += 1
        col += 1
        # M5
        U[0*3+0][col] += 1; U[0*3+1][col] += 1
        V[1*5+1][col] += 1
        W[1*2+0][col] += 1
        col += 1
        # M6
        U[1*3+1][col] += 1
        V[0*5+0][col] += 1; V[1*5+0][col] += -1
        W[1*2+1][col] += 1
        col += 1
        # M7
        U[0*3+1][col] += 1; U[0*3+0][col] += -1
        V[0*5+0][col] += 1; V[1*5+1][col] += 1
        W[0*2+0][col] += 1; W[1*2+1][col] += 1
        col += 1

    # Block 1: A1 (cols 0,1) * B11 (rows 0,1, cols 0,1)
    add_strassen(0, 0)
    
    # Block 2: A1 (cols 0,1) * B12 (rows 0,1, cols 2,3)
    # Need to adjust V indices for k_off=2
    # My add_strassen assumes k starts at 0.
    # I should generalize or just write it out.
    # For now, I'll write a more generic helper or just duplicate code with offsets.
    
    # Let's re-implement with offsets to be safe.
    def add_strassen_gen(j_off, k_off):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[0*3+j_off][col] += 1; U[1*3+j_off+1][col] += 1
        V[j_off*5+k_off][col] += 1; V[j_off+1*5+k_off+1][col] += 1
        W[k_off*2+0][col] += 1; W[k_off+1*2+1][col] += 1
        col += 1
        # M2: (A21+A22)B12
        U[1*3+j_off][col] += 1; U[1*3+j_off+1][col] += 1
        V[j_off*5+k_off+1][col] += 1
        W[k_off+1*2+0][col] += 1
        col += 1
        # M3: A11(B21-B11)
        U[0*3+j_off][col] += 1
        V[j_off+1*5+k_off][col] += 1; V[j_off*5+k_off][col] += -1
        W[k_off*2+1][col] += 1
        col += 1
        # M4: (A11-A21)(B22+B12)
        U[0*3+j_off][col] += 1; U[1*3+j_off][col] += -1
        V[j_off+1*5+k_off+1][col] += 1; V[j_off*5+k_off+1][col] += 1
        W[k_off*2+0][col] += 1; W[k_off+1*2+0][col] += -1; W[k_off+1*2+1][col] += 1
        col += 1
        # M5: (A11+A12)B22
        U[0*3+j_off][col] += 1; U[0*3+j_off+1][col] += 1
        V[j_off+1*5+k_off+1][col] += 1
        W[k_off*2+1][col] += 1
        col += 1
        # M6: A22(B11-B21)
        U[1*3+j_off+1][col] += 1
        V[j_off*5+k_off][col] += 1; V[j_off+1*5+k_off][col] += -1
        W[k_off+1*2+1][col] += 1
        col += 1
        # M7: (A12-A11)(B11+B22)
        U[0*3+j_off+1][col] += 1; U[0*3+j_off][col] += -1
        V[j_off*5+k_off][col] += 1; V[j_off+1*5+k_off+1][col] += 1
        W[k_off*2+0][col] += 1; W[k_off+1*2+1][col] += 1
        col += 1

    # Reset col
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # 1. A1 (j=0,1) * B11 (j=0,1, k=0,1)
    add_strassen_gen(0, 0)
    
    # 2. A1 (j=0,1) * B12 (j=0,1, k=2,3)
    add_strassen_gen(0, 2)
    
    # 3. A1 (j=0,1) * B13 (j=0,1, k=4)
    # Schoolbook 2x2x1
    # C_ik = A_i0 B_0k + A_i1 B_1k
    for i in range(2):
        for j in range(2):
            U[i*3+j][col] = 1
            V[j*5+4][col] = 1
            W[4*2+i][col] = 1
            col += 1
            
    # 4. A2 (j=2) * B2 (j=2, k=0..4)
    # A2 is 2x1 (col 2). B2 is 1x5 (row 2).
    # C_ik = A_i2 B_2k
    for i in range(2):
        for k in range(5):
            U[i*3+2][col] = 1
            V[2*5+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
