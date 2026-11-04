import numpy as np

def construct():
    n, m, p = 2, 5, 5
    rank = 46
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    # A_idx: (i1, i2, j1, j2) -> A[i, j]
    # B_idx: (j1, j2, k1, k2) -> B[j, k]
    # W_idx: (k1, k2, i1, i2) -> W[k, i]
    def add_strassen(a_i1, a_i2, a_j1, a_j2, b_j1, b_j2, b_k1, b_k2, w_k1, w_k2, w_i1, w_i2):
        nonlocal col
        # Strassen steps
        # M1 = A11 * (B11 + B22)
        U[a_i1*m + a_j1, col] = 1
        V[b_j1*p + b_k1, col] = 1
        V[b_j2*p + b_k2, col] = 1
        W[b_k1*n + w_i1, col] = 1 # Wait, W index is k*n+i.
        # Let's trace W indices carefully.
        # Result C is 2x2.
        # C11 = M5 + M4 - M2 + M6
        # C12 = M1 - M3 + M7
        # C21 = M3 + M4 - M5 - M7
        # C22 = M1 - M2 + M5
        
        # We need to add 7 terms.
        
        # Term 1: M1 = A11 * (B11 + B22)
        # U: A11
        # V: B11 + B22
        # W: contributes to C12 (+1) and C22 (-1)
        U[a_i1*m + a_j1, col] = 1
        V[b_j1*p + b_k1, col] = 1
        V[b_j2*p + b_k2, col] = 1
        W[b_k1*n + w_i2, col] = 1 # C12 is row 0, col 1? 
        # Let's map local C indices to global W indices.
        # Local C row 0, col 0 -> Global W index b_k1*n + w_i1
        # Local C row 0, col 1 -> Global W index b_k2*n + w_i1
        # Local C row 1, col 0 -> Global W index b_k1*n + w_i2
        # Local C row 1, col 1 -> Global W index b_k2*n + w_i2
        
        # M1 contributes to C12 (local 0,1) and C22 (local 1,1)
        # C12: W[b_k2*n + w_i1]
        # C22: W[b_k2*n + w_i2]
        W[b_k2*n + w_i1, col] = 1
        W[b_k2*n + w_i2, col] = -1
        col += 1
        
        # Term 2: M2 = (A11 + A12) * B22
        U[a_i1*m + a_j1, col] = 1
        U[a_i1*m + a_j2, col] = 1
        V[b_j2*p + b_k2, col] = 1
        # Contributes to C11 (-1) and C22 (-1)
        W[b_k1*n + w_i1, col] = -1
        W[b_k2*n + w_i2, col] = -1
        col += 1
        
        # Term 3: M3 = A22 * (B11 + B12)
        U[a_i2*m + a_j2, col] = 1
        V[b_j2*p + b_k1, col] = 1
        V[b_j2*p + b_k2, col] = 1
        # Contributes to C12 (-1) and C21 (+1)
        W[b_k1*n + w_i2, col] = 1
        W[b_k2*n + w_i1, col] = -1
        col += 1
        
        # Term 4: M4 = A21 * (B21 + B22)
        U[a_i2*m + a_j1, col] = 1
        V[b_j1*p + b_k2, col] = 1
        V[b_j2*p + b_k2, col] = 1
        # Contributes to C11 (+1) and C21 (+1)
        W[b_k1*n + w_i1, col] = 1
        W[b_k1*n + w_i2, col] = 1
        col += 1
        
        # Term 5: M5 = (A11 + A22) * B11
        U[a_i1*m + a_j1, col] = 1
        U[a_i2*m + a_j2, col] = 1
        V[b_j1*p + b_k1, col] = 1
        # Contributes to C11 (+1) and C21 (-1)
        W[b_k1*n + w_i1, col] = 1
        W[b_k1*n + w_i2, col] = -1
        col += 1
        
        # Term 6: M6 = (A12 - A22) * (B21 + B22)
        U[a_i1*m + a_j2, col] = 1
        U[a_i2*m + a_j2, col] = -1
        V[b_j1*p + b_k2, col] = 1
        V[b_j2*p + b_k2, col] = 1
        # Contributes to C11 (+1)
        W[b_k1*n + w_i1, col] = 1
        col += 1
        
        # Term 7: M7 = (A11 - A21) * (B11 + B12)
        U[a_i1*m + a_j1, col] = 1
        U[a_i2*m + a_j1, col] = -1
        V[b_j1*p + b_k1, col] = 1
        V[b_j2*p + b_k1, col] = 1
        # Contributes to C12 (+1) and C21 (-1)
        W[b_k2*n + w_i1, col] = 1
        W[b_k1*n + w_i2, col] = -1
        col += 1

    # Helper to add Schoolbook 2x2x1 block
    def add_school_221(a_i1, a_i2, a_j1, a_j2, b_j1, b_j2, b_k1, w_k1, w_i1, w_i2):
        nonlocal col
        # C11 = A11 B11 + A12 B21
        # C21 = A21 B11 + A22 B21
        
        # Term 1: A11 B11
        U[a_i1*m + a_j1, col] = 1
        V[b_j1*p + b_k1, col] = 1
        W[b_k1*n + w_i1, col] = 1
        col += 1
        
        # Term 2: A12 B21
        U[a_i1*m + a_j2, col] = 1
        V[b_j2*p + b_k1, col] = 1
        W[b_k1*n + w_i1, col] = 1
        col += 1
        
        # Term 3: A21 B11
        U[a_i2*m + a_j1, col] = 1
        V[b_j1*p + b_k1, col] = 1
        W[b_k1*n + w_i2, col] = 1
        col += 1
        
        # Term 4: A22 B21
        U[a_i2*m + a_j2, col] = 1
        V[b_j2*p + b_k1, col] = 1
        W[b_k1*n + w_i2, col] = 1
        col += 1

    # Helper to add Outer Product 2x1x5 block
    def add_outer(a_i1, a_i2, a_j1, b_j1, b_k1, b_k2, b_k3, b_k4, b_k5, w_k1, w_k2, w_k3, w_k4, w_k5, w_i1, w_i2):
        nonlocal col
        # C1j = A11 B1j
        # C2j = A21 B1j
        
        for k_idx in range(5):
            b_k = [b_k1, b_k2, b_k3, b_k4, b_k5][k_idx]
            w_k = [w_k1, w_k2, w_k3, w_k4, w_k5][k_idx]
            
            # Term 1: A11 B1j
            U[a_i1*m + a_j1, col] = 1
            V[b_j1*p + b_k, col] = 1
            W[w_k*n + w_i1, col] = 1
            col += 1
            
            # Term 2: A21 B1j
            U[a_i2*m + a_j1, col] = 1
            V[b_j1*p + b_k, col] = 1
            W[w_k*n + w_i2, col] = 1
            col += 1

    # Block 1: A1 (0,1; 0,1) * B1_1 (0,1; 0,1)
    # A: i=0,1; j=0,1
    # B: j=0,1; k=0,1
    # W: k=0,1; i=0,1
    add_strassen(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    
    # Block 2: A1 (0,1; 0,1) * B1_2 (0,1; 2,3)
    # A: i=0,1; j=0,1
    # B: j=0,1; k=2,3
    # W: k=2,3; i=0,1
    add_strassen(0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 0, 1)
    
    # Block 3: A1 (0,1; 0,1) * B1_3 (0,1; 4)
    # A: i=0,1; j=0,1
    # B: j=0,1; k=4
    # W: k=4; i=0,1
    add_school_221(0, 1, 0, 1, 0, 1, 4, 4, 0, 1)
    
    # Block 4: A2_1 (0,1; 2,3) * B2_1 (2,3; 0,1)
    # A: i=0,1; j=2,3
    # B: j=2,3; k=0,1
    # W: k=0,1; i=0,1
    add_strassen(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1)
    
    # Block 5: A2_1 (0,1; 2,3) * B2_1 (2,3; 2,3)
    # A: i=0,1; j=2,3
    # B: j=2,3; k=2,3
    # W: k=2,3; i=0,1
    add_strassen(0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 0, 1)
    
    # Block 6: A2_1 (0,1; 2,3) * B2_1 (2,3; 4)
    # A: i=0,1; j=2,3
    # B: j=2,3; k=4
    # W: k=4; i=0,1
    add_school_221(0, 1, 2, 3, 2, 3, 4, 4, 0, 1)
    
    # Block 7: A2_2 (0,1; 4) * B2_2 (4; 0,1,2,3,4)
    # A: i=0,1; j=4
    # B: j=4; k=0,1,2,3,4
    # W: k=0,1,2,3,4; i=0,1
    add_outer(0, 1, 4, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
