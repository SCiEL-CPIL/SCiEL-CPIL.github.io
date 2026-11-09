import numpy as np

def construct():
    n, m, p = 2, 4, 7
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block
    # x_base: index offset in U for A part (2x2)
    # y_base: index offset in V for B part (2x2)
    # w_base: index offset in W for C part (2x2)
    # col: current column index in U, V, W
    def add_strassen(x_base, y_base, w_base, col):
        # M1
        U[x_base + 0][col] += 1 # A00
        U[x_base + 2][col] += 1 # A11
        V[y_base + 0][col] += 1 # B00
        V[y_base + 1][col] += 1 # B01
        W[w_base + 0][col] += 1 # C00
        W[w_base + 1][col] += -1 # C01
        W[w_base + 2][col] += -1 # C10
        W[w_base + 3][col] += 1 # C11
        
        # M2
        U[x_base + 2][col+1] += 1 # A10
        U[x_base + 3][col+1] += 1 # A11
        V[y_base + 0][col+1] += 1 # B00
        W[w_base + 2][col+1] += -1 # C10
        W[w_base + 3][col+1] += 1 # C11
        
        # M3
        U[x_base + 0][col+2] += 1 # A00
        V[y_base + 1][col+2] += 1 # B01
        V[y_base + 3][col+2] += -1 # B11
        W[w_base + 1][col+2] += 1 # C01
        W[w_base + 3][col+2] += -1 # C11
        
        # M4
        U[x_base + 3][col+3] += 1 # A11
        V[y_base + 2][col+3] += 1 # B10
        V[y_base + 0][col+3] += -1 # B00
        W[w_base + 0][col+3] += 1 # C00
        W[w_base + 2][col+3] += 1 # C10
        
        # M5
        U[x_base + 0][col+4] += 1 # A00
        U[x_base + 1][col+4] += 1 # A01
        V[y_base + 3][col+4] += 1 # B11
        W[w_base + 0][col+4] += -1 # C00
        W[w_base + 1][col+4] += 1 # C01
        
        # M6
        U[x_base + 2][col+5] += 1 # A10
        U[x_base + 0][col+5] += -1 # A00
        V[y_base + 0][col+5] += 1 # B00
        V[y_base + 1][col+5] += 1 # B01
        W[w_base + 3][col+5] += 1 # C11
        
        # M7
        U[x_base + 1][col+6] += 1 # A01
        U[x_base + 3][col+6] += -1 # A11
        V[y_base + 2][col+6] += 1 # B10
        V[y_base + 3][col+6] += 1 # B11
        W[w_base + 0][col+6] += 1 # C00
        W[w_base + 2][col+6] += -1 # C10
        
        return col + 7

    # Helper to add a schoolbook 2x2x1 block
    # x_base: index offset in U for A part (2x2)
    # y_base: index offset in V for B part (2x1)
    # w_base: index offset in W for C part (2x1)
    # col: current column index
    def add_schoolbook_2x2x1(x_base, y_base, w_base, col):
        # C00 = A00*B00 + A01*B10
        # C10 = A10*B00 + A11*B10
        # This is 2x2 times 2x1.
        # We need 4 multiplications.
        
        # M1: A00 * B00
        U[x_base + 0][col] += 1
        V[y_base + 0][col] += 1
        W[w_base + 0][col] += 1
        
        # M2: A01 * B10
        U[x_base + 1][col+1] += 1
        V[y_base + 1][col+1] += 1
        W[w_base + 0][col+1] += 1
        
        # M3: A10 * B00
        U[x_base + 2][col+2] += 1
        V[y_base + 0][col+2] += 1
        W[w_base + 1][col+2] += 1
        
        # M4: A11 * B10
        U[x_base + 3][col+3] += 1
        V[y_base + 1][col+3] += 1
        W[w_base + 1][col+3] += 1
        
        return col + 4

    col = 0
    
    # C1 = A1 B1
    # A1 is A[0:2, 0:2]. U indices: 0,1,2,3 (i*m+j)
    # B1 is B[0:2, :]. 
    # B11 is B[0:2, 0:2]. V indices: 0,1,2,3 (j*p+k)
    # B12 is B[0:2, 2:4]. V indices: 0,1, 7,8? No.
    # V index = j*p + k.
    # j=0, k=0..1 -> 0, 1
    # j=1, k=0..1 -> 7, 8
    # So B11 V base is 0.
    # B12 V base is 2*7 + 0 = 14? No.
    # B12 is cols 2,3.
    # j=0, k=2,3 -> 2, 3
    # j=1, k=2,3 -> 9, 10
    # This is not a contiguous block in V if we just use base.
    # My add_strassen assumes contiguous 2x2 block in V.
    # V layout: row j, col k.
    # For B11 (j=0,1; k=0,1):
    # (0,0)->0, (0,1)->1, (1,0)->7, (1,1)->8.
    # Not contiguous 0,1,2,3.
    # So I need to pass explicit indices or handle the mapping.
    
    # Let's redefine add_strassen to take lists of indices.
    
    def add_strassen_explicit(U_idx, V_idx, W_idx, col):
        # U_idx: [u00, u01, u10, u11]
        # V_idx: [v00, v01, v10, v11]
        # W_idx: [w00, w01, w10, w11]
        
        # M1
        U[U_idx[0]][col] += 1
        U[U_idx[3]][col] += 1
        V[V_idx[0]][col] += 1
        V[V_idx[1]][col] += 1
        W[W_idx[0]][col] += 1
        W[W_idx[1]][col] += -1
        W[W_idx[2]][col] += -1
        W[W_idx[3]][col] += 1
        
        # M2
        U[U_idx[2]][col+1] += 1
        U[U_idx[3]][col+1] += 1
        V[V_idx[0]][col+1] += 1
        W[W_idx[2]][col+1] += -1
        W[W_idx[3]][col+1] += 1
        
        # M3
        U[U_idx[0]][col+2] += 1
        V[V_idx[1]][col+2] += 1
        V[V_idx[3]][col+2] += -1
        W[W_idx[1]][col+2] += 1
        W[W_idx[3]][col+2] += -1
        
        # M4
        U[U_idx[3]][col+3] += 1
        V[V_idx[2]][col+3] += 1
        V[V_idx[0]][col+3] += -1
        W[W_idx[0]][col+3] += 1
        W[W_idx[2]][col+3] += 1
        
        # M5
        U[U_idx[0]][col+4] += 1
        U[U_idx[1]][col+4] += 1
        V[V_idx[3]][col+4] += 1
        W[W_idx[0]][col+4] += -1
        W[W_idx[1]][col+4] += 1
        
        # M6
        U[U_idx[2]][col+5] += 1
        U[U_idx[0]][col+5] += -1
        V[V_idx[0]][col+5] += 1
        V[V_idx[1]][col+5] += 1
        W[W_idx[3]][col+5] += 1
        
        # M7
        U[U_idx[1]][col+6] += 1
        U[U_idx[3]][col+6] += -1
        V[V_idx[2]][col+6] += 1
        V[V_idx[3]][col+6] += 1
        W[W_idx[0]][col+6] += 1
        W[W_idx[2]][col+6] += -1
        
        return col + 7

    def add_schoolbook_explicit(U_idx, V_idx, W_idx, col):
        # U_idx: [u00, u01, u10, u11]
        # V_idx: [v00, v10] (2x1)
        # W_idx: [w00, w10] (2x1)
        
        # M1: A00 * B00
        U[U_idx[0]][col] += 1
        V[V_idx[0]][col] += 1
        W[W_idx[0]][col] += 1
        
        # M2: A01 * B10
        U[U_idx[1]][col+1] += 1
        V[V_idx[1]][col+1] += 1
        W[W_idx[0]][col+1] += 1
        
        # M3: A10 * B00
        U[U_idx[2]][col+2] += 1
        V[V_idx[0]][col+2] += 1
        W[W_idx[1]][col+2] += 1
        
        # M4: A11 * B10
        U[U_idx[3]][col+3] += 1
        V[V_idx[1]][col+3] += 1
        W[W_idx[1]][col+3] += 1
        
        return col + 4

    # Indices for A1 (2x2) in U
    # U index = i*m + j = i*4 + j
    U_A1 = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # 0, 1, 4, 5
    
    # Indices for A2 (2x2) in U
    U_A2 = [0*4+2, 0*4+3, 1*4+2, 1*4+3] # 2, 3, 6, 7
    
    # Indices for W (2x7)
    # W index = k*n + i = k*2 + i
    # For C1 (cols 0-6 of B1)
    # We need W indices for each 2x2 or 2x1 block of C.
    # C11 (cols 0,1): W_idx = [0*2+0, 0*2+1, 1*2+0, 1*2+1] = [0, 1, 2, 3]
    # C12 (cols 2,3): W_idx = [2*2+0, 2*2+1, 3*2+0, 3*2+1] = [4, 5, 6, 7]
    # C13 (cols 4,5): W_idx = [4*2+0, 4*2+1, 5*2+0, 5*2+1] = [8, 9, 10, 11]
    # C14 (col 6): W_idx = [6*2+0, 6*2+1] = [12, 13]
    
    # Indices for V (4x7)
    # V index = j*p + k = j*7 + k
    # B11 (j=0,1; k=0,1): V_idx = [0*7+0, 0*7+1, 1*7+0, 1*7+1] = [0, 1, 7, 8]
    # B12 (j=0,1; k=2,3): V_idx = [0*7+2, 0*7+3, 1*7+2, 1*7+3] = [2, 3, 9, 10]
    # B13 (j=0,1; k=4,5): V_idx = [0*7+4, 0*7+5, 1*7+4, 1*7+5] = [4, 5, 11, 12]
    # B14 (j=0,1; k=6): V_idx = [0*7+6, 1*7+6] = [6, 13]
    
    # B21 (j=2,3; k=0,1): V_idx = [2*7+0, 2*7+1, 3*7+0, 3*7+1] = [14, 15, 21, 22]
    # B22 (j=2,3; k=2,3): V_idx = [2*7+2, 2*7+3, 3*7+2, 3*7+3] = [16, 17, 23, 24]
    # B23 (j=2,3; k=4,5): V_idx = [2*7+4, 2*7+5, 3*7+4, 3*7+5] = [18, 19, 25, 26]
    # B24 (j=2,3; k=6): V_idx = [2*7+6, 3*7+6] = [20, 27]
    
    # W indices are the same for C1 and C2 because they add to the same C.
    # Wait, C1 and C2 are added. So W columns for C1 and C2 are distinct columns in the rank dimension.
    # But the W indices (rows) are the same.
    
    # C1 Part
    col = add_strassen_explicit(U_A1, [0, 1, 7, 8], [0, 1, 2, 3], col)
    col = add_strassen_explicit(U_A1, [2, 3, 9, 10], [4, 5, 6, 7], col)
    col = add_strassen_explicit(U_A1, [4, 5, 11, 12], [8, 9, 10, 11], col)
    col = add_schoolbook_explicit(U_A1, [6, 13], [12, 13], col)
    
    # C2 Part
    col = add_strassen_explicit(U_A2, [14, 15, 21, 22], [0, 1, 2, 3], col)
    col = add_strassen_explicit(U_A2, [16, 17, 23, 24], [4, 5, 6, 7], col)
    col = add_strassen_explicit(U_A2, [18, 19, 25, 26], [8, 9, 10, 11], col)
    col = add_schoolbook_explicit(U_A2, [20, 27], [12, 13], col)
    
    return {"U": U, "V": V, "W": W}
