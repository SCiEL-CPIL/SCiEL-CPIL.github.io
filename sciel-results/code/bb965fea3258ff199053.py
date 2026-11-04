import numpy as np

def construct():
    n, m, p = 2, 4, 7
    R = 52
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    r = 0
    
    # Helper to add Strassen M_2,2,2
    # A_indices: [a11, a12, a21, a22] indices in U
    # B_indices: [b11, b12, b21, b22] indices in V
    # C_indices: [c11, c12, c21, c22] indices in W
    def add_strassen(A_idx, B_idx, C_idx):
        nonlocal r
        # m1 = a11(b11+b22)
        U[A_idx[0], r] = 1
        V[B_idx[0], r] = 1
        V[B_idx[3], r] = 1
        W[C_idx[0], r] = 1
        W[C_idx[3], r] = -1
        r += 1
        # m2 = (a11+a21)b22
        U[A_idx[0], r] = 1
        U[A_idx[2], r] = 1
        V[B_idx[3], r] = 1
        W[C_idx[2], r] = 1
        W[C_idx[3], r] = -1
        r += 1
        # m3 = (a12+a22)b11
        U[A_idx[1], r] = 1
        U[A_idx[3], r] = 1
        V[B_idx[0], r] = 1
        W[C_idx[1], r] = 1
        W[C_idx[3], r] = 1
        r += 1
        # m4 = a22(b21+b11)
        U[A_idx[3], r] = 1
        V[B_idx[2], r] = 1
        V[B_idx[0], r] = 1
        W[C_idx[0], r] = 1
        W[C_idx[2], r] = 1
        r += 1
        # m5 = (a11+a12)b22
        U[A_idx[0], r] = 1
        U[A_idx[1], r] = 1
        V[B_idx[3], r] = 1
        W[C_idx[0], r] = -1
        W[C_idx[1], r] = 1
        r += 1
        # m6 = (a21-a11)(b11+b12)
        U[A_idx[2], r] = 1
        U[A_idx[0], r] = -1
        V[B_idx[0], r] = 1
        V[B_idx[1], r] = 1
        W[C_idx[3], r] = 1
        r += 1
        # m7 = (a12-a22)(b21+b22)
        U[A_idx[1], r] = 1
        U[A_idx[3], r] = -1
        V[B_idx[2], r] = 1
        V[B_idx[3], r] = 1
        W[C_idx[0], r] = 1
        W[C_idx[1], r] = 1
        r += 1

    # Helper to add Schoolbook M_2,2,1
    # A_indices: [a11, a12, a21, a22] indices in U
    # B_indices: [b1, b2] indices in V
    # C_indices: [c1, c2] indices in W
    def add_schoolbook_221(A_idx, B_idx, C_idx):
        nonlocal r
        # c1 = a11*b1 + a12*b2
        # c2 = a21*b1 + a22*b2
        U[A_idx[0], r] = 1
        V[B_idx[0], r] = 1
        W[C_idx[0], r] = 1
        r += 1
        U[A_idx[1], r] = 1
        V[B_idx[1], r] = 1
        W[C_idx[0], r] = 1
        r += 1
        U[A_idx[2], r] = 1
        V[B_idx[0], r] = 1
        W[C_idx[1], r] = 1
        r += 1
        U[A_idx[3], r] = 1
        V[B_idx[1], r] = 1
        W[C_idx[1], r] = 1
        r += 1

    # A indices for left half (cols 0,1)
    A_L = [0, 1, 4, 5] # a11, a12, a21, a22
    # A indices for right half (cols 2,3)
    A_R = [2, 3, 6, 7]
    
    # Process B cols 0-3 (M_2,4,4 part)
    # Block 1: cols 0,1
    # B_T (rows 0,1) for cols 0,1
    B_T_01 = [0*7+0, 0*7+1, 1*7+0, 1*7+1]
    B_B_01 = [2*7+0, 2*7+1, 3*7+0, 3*7+1]
    C_01 = [0*2+0, 0*2+1, 1*2+0, 1*2+1]
    
    add_strassen(A_L, B_T_01, C_01)
    add_strassen(A_R, B_B_01, C_01)
    
    # Block 2: cols 2,3
    B_T_23 = [0*7+2, 0*7+3, 1*7+2, 1*7+3]
    B_B_23 = [2*7+2, 2*7+3, 3*7+2, 3*7+3]
    C_23 = [2*2+0, 2*2+1, 3*2+0, 3*2+1]
    
    add_strassen(A_L, B_T_23, C_23)
    add_strassen(A_R, B_B_23, C_23)
    
    # Process B cols 4-6 (M_2,4,3 part)
    for c in range(4, 7):
        # B_T (rows 0,1) for col c
        B_T_c = [0*7+c, 1*7+c]
        B_B_c = [2*7+c, 3*7+c]
        C_c = [c*2+0, c*2+1]
        
        add_schoolbook_221(A_L, B_T_c, C_c)
        add_schoolbook_221(A_R, B_B_c, C_c)
        
    return {"U": U, "V": V, "W": W}
