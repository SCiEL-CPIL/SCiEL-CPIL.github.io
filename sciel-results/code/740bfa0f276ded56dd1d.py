"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # U shape (6, R), V shape (30, R), W shape (20, R)
    
    # We will build the matrices U, V, W column by column (each column is a rank-1 term)
    # Rank 55
    # 35 terms from Strassen on 5 blocks of (2x2) * (2x2)
    # 20 terms from schoolbook on (2x1) * (1x10)
    
    R = 55
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for A[0:2, 0:2] * B[0:2, :]
    # B is split into 5 blocks of 2 columns
    # A indices: i in 0..1, j in 0..1
    # B indices: j in 0..1, k in 0..9
    # C indices: i in 0..1, k in 0..9
    
    # Strassen intermediates for 2x2 * 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01+B11)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the linear forms for A, B and coefficients for C
    # A forms (indices i, j for i,j in 0,1)
    # B forms (indices j, k_local for j in 0,1, k_local in 0,1)
    # C coeffs (indices i, k_local for i in 0,1, k_local in 0,1)
    
    # s=0 corresponds to M1, etc.
    
    # M1
    A1 = {(0,0): 1, (1,1): 1}
    B1 = {(0,0): 1, (1,1): 1}
    C1 = {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1} # Wait, C00 and C11 get +1
    
    # M2
    A2 = {(1,0): 1, (1,1): 1}
    B2 = {(0,0): 1}
    C2 = {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1} # C10 +1, C11 -1
    
    # M3
    A3 = {(0,0): 1}
    B3 = {(1,0): 1, (1,1): -1}
    C3 = {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1} # C01 +1, C11 +1
    
    # M4
    A4 = {(1,1): 1}
    B4 = {(0,1): 1, (1,1): 1}
    C4 = {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0} # C00 +1, C10 +1
    
    # M5
    A5 = {(0,0): 1, (1,0): 1}
    B5 = {(1,1): 1}
    C5 = {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0} # C00 -1, C01 +1
    
    # M6
    A6 = {(1,1): 1, (0,0): -1}
    B6 = {(0,0): 1, (0,1): 1}
    C6 = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1} # C11 +1
    
    # M7
    A7 = {(0,1): 1, (1,1): -1}
    B7 = {(1,0): 1, (1,1): 1}
    C7 = {(0,0): -1, (0,1): 0, (1,0): 0, (1,1): 0} # C00 -1
    
    strassen_steps = [
        (A1, B1, C1),
        (A2, B2, C2),
        (A3, B3, C3),
        (A4, B4, C4),
        (A5, B5, C5),
        (A6, B6, C6),
        (A7, B7, C7)
    ]
    
    for block in range(5):
        # k offset for this block
        k_offset = block * 2
        for A_form, B_form, C_form in strassen_steps:
            # U: n*m = 6
            # i in 0..1, j in 0..1
            for (i, j), val in A_form.items():
                U[i * m + j][col] = val
            
            # V: m*p = 30
            # j in 0..1, k_local in 0..1 -> k = k_offset + k_local
            for (j, k_local), val in B_form.items():
                k = k_offset + k_local
                V[j * p + k][col] = val
            
            # W: p*n = 20
            # k_local in 0..1, i in 0..1 -> k = k_offset + k_local
            for (i, k_local), val in C_form.items():
                k = k_offset + k_local
                W[k * n + i][col] = val
            
            col += 1
            
    # Schoolbook for A[0:2, 2:3] * B[2:3, 0:10]
    # A indices: i in 0..1, j=2
    # B indices: j=2, k in 0..9
    # C indices: i in 0..1, k in 0..9
    
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
