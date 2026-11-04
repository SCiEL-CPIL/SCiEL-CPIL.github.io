import numpy as np

def construct():
    n, m, p = 3, 4, 4
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2
    # Local indices i,j,k in 0,1
    # U_idx = i*m + j
    # V_idx = j*p + k
    # W_idx = k*n + i
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        # ...
        # We need to add 7 terms
        
        # Term 1: (A00+A11)(B00+B11)
        # A: (0,0), (1,1) -> +1
        # B: (0,0), (1,1) -> +1
        # C: (0,0)+, (0,1)+, (1,0)+, (1,1)-
        
        # It's easier to define the 7 Strassen mults explicitly
        
        # M1
        U[i_off*4 + 0][col] = 1
        U[i_off*4 + 1][col] = 1
        V[j_off*4 + 0][col] = 1
        V[j_off*4 + 1][col] = 1
        W[k_off*3 + 0][col] = 1
        W[k_off*3 + 1][col] = 1
        W[k_off*3 + 3][col] = 1
        W[k_off*3 + 4][col] = -1
        col += 1
        
        # M2 = (A10+A11)B00
        U[i_off*4 + 1][col] = 1 # A10
        U[i_off*4 + 2][col] = 1 # A11 (j=1)
        V[j_off*4 + 0][col] = 1 # B00
        W[k_off*3 + 3][col] = 1 # C10
        W[k_off*3 + 4][col] = -1 # C11
        col += 1
        
        # M3 = A00(B01-B11)
        U[i_off*4 + 0][col] = 1 # A00
        V[j_off*4 + 1][col] = 1 # B01
        V[j_off*4 + 3][col] = -1 # B11
        W[k_off*3 + 1][col] = 1 # C01
        W[k_off*3 + 4][col] = 1 # C11
        col += 1
        
        # M4 = A11(B10-B00)
        U[i_off*4 + 2][col] = 1 # A11
        V[j_off*4 + 2][col] = 1 # B10
        V[j_off*4 + 0][col] = -1 # B00
        W[k_off*3 + 3][col] = 1 # C10
        W[k_off*3 + 4][col] = -1 # C11
        col += 1
        
        # M5 = (A00+A01)B11
        U[i_off*4 + 0][col] = 1 # A00
        U[i_off*4 + 1][col] = 1 # A01
        V[j_off*4 + 3][col] = 1 # B11
        W[k_off*3 + 3][col] = -1 # C10
        W[k_off*3 + 4][col] = 1 # C11
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[i_off*4 + 1][col] = 1 # A10
        U[i_off*4 + 0][col] = -1 # A00
        V[j_off*4 + 0][col] = 1 # B00
        V[j_off*4 + 1][col] = 1 # B01
        W[k_off*3 + 4][col] = 1 # C11
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i_off*4 + 1][col] = 1 # A01
        U[i_off*4 + 2][col] = -1 # A11
        V[j_off*4 + 2][col] = 1 # B10
        V[j_off*4 + 3][col] = 1 # B11
        W[k_off*3 + 4][col] = 1 # C11
        col += 1

    # C_top_left: i=0,1; j=0..3; k=0,1
    add_strassen(0, 0, 0) # A_L, B_T
    add_strassen(0, 2, 0) # A_R, B_B
    
    # C_top_right: i=0,1; j=0..3; k=2,3
    add_strassen(0, 0, 2) # A_L, B_T'
    add_strassen(0, 2, 2) # A_R, B_B'
    
    # C_bottom: i=2; j=0..3; k=0..3
    for j in range(4):
        for k in range(4):
            U[2*4 + j][col] = 1
            V[j*4 + k][col] = 1
            W[k*3 + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
