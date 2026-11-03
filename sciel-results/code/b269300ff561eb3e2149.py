import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will use rank 52
    # 14 for C1 (2x2), 14 for C2 (2x2), 24 for C3 (2x3)
    rank = 52
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add Strassen terms for 2x2 * 2x2
    # A is 2x2, B is 2x2
    # Returns list of (u_vec, v_vec, w_vec) for the 7 terms
    # u_vec is in R^4 (A), v_vec in R^4 (B), w_vec in R^4 (C)
    def strassen_2x2(a_idx, b_idx, c_idx):
        # a_idx: indices in A (flattened 2x2)
        # b_idx: indices in B (flattened 2x2)
        # c_idx: indices in C (flattened 2x2)
        terms = []
        
        # M1 = (A11 + A22)(B11 + B22)
        # u: A11, A22
        # v: B11, B22
        # w: C11, C12, C21, C22
        # C11 += M1, C12 -= M1, C21 -= M1, C22 += M1
        
        # We will generate the vectors for the full 8x28x14 space
        # This is complex. It is easier to just iterate over the 7 terms and set the values.
        
        # Term 1: (A11 + A22)(B11 + B22)
        # u: A11=1, A22=1
        # v: B11=1, B22=1
        # w: C11=1, C12=-1, C21=-1, C22=1
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        
        # Map local 2x2 indices to global indices
        # A11 is A[0, a_idx[0]], A12 is A[0, a_idx[1]]
        # A21 is A[1, a_idx[0]], A22 is A[1, a_idx[1]]
        
        # We need to pass the global offsets.
        pass

    # It is better to explicitly write out the 52 terms.
    
    # Block 1: C1 = A1 B11 + A2 B21
    # A1 is A[:, 0:2], A2 is A[:, 2:4]
    # B11 is B[0:2, 0:2], B21 is B[2:4, 0:2]
    
    # Strassen for A1 B11
    # A1 indices: 0,1,2,3 (flat 2x2)
    # B11 indices: 0,1,4,5 (flat 4x7, rows 0,1)
    # C1 indices: 0,1,7,8 (flat 2x7, cols 0,1)
    
    # Strassen for A2 B21
    # A2 indices: 4,5,6,7
    # B21 indices: 2,3,6,7
    # C1 indices: 0,1,7,8
    
    # We will use a helper function to add a term
    def add_term(u_val, v_val, w_val):
        nonlocal col
        U[0, col] = u_val[0]
        U[1, col] = u_val[1]
        U[2, col] = u_val[2]
        U[3, col] = u_val[3]
        U[4, col] = u_val[4]
        U[5, col] = u_val[5]
        U[6, col] = u_val[6]
        U[7, col] = u_val[7]
        
        for i in range(m*p):
            V[i, col] = v_val[i]
            
        for i in range(p*n):
            W[i, col] = w_val[i]
            
        col += 1

    # This is too slow to write manually.
    # I will generate the matrices using loops.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Function to add Strassen terms for a specific 2x2 block
    # a_off: offset of A block (0 or 2)
    # b_off_row: offset of B block row (0 or 2)
    # b_off_col: offset of B block col (0, 2, 4)
    # c_off_col: offset of C block col (0, 2, 4)
    # is_2x3: if True, B is 2x3, C is 2x3, use schoolbook
    
    def add_strassen_2x2(a_off, b_off_r, b_off_c, c_off_c):
        nonlocal col
        # A indices: a_off, a_off+1, a_off+2, a_off+3
        # B indices: b_off_r*m + b_off_c, ...
        # C indices: c_off_c, ...
        
        # M1
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off] = 1; u[a_off+3] = 1
        v[b_off_r*m + b_off_c] = 1; v[b_off_r*m + b_off_c + 1] = 1 # B11, B12? No, B is 2x2.
        # B11 is at (b_off_r, b_off_c)
        # B12 is at (b_off_r, b_off_c+1)
        # B21 is at (b_off_r+1, b_off_c)
        # B22 is at (b_off_r+1, b_off_c+1)
        
        # Wait, B is 4x7.
        # B11 (2x2) is at rows b_off_r, b_off_r+1 and cols b_off_c, b_off_c+1.
        
        v[b_off_r*m + b_off_c] = 1
        v[(b_off_r+1)*m + (b_off_c+1)] = 1
        
        # C11, C12, C21, C22
        # C is 2x7.
        # C11 at (0, c_off_c)
        # C12 at (0, c_off_c+1)
        # C21 at (1, c_off_c)
        # C22 at (1, c_off_c+1)
        
        w[0*n + c_off_c] = 1
        w[0*n + c_off_c + 1] = -1
        w[1*n + c_off_c] = -1
        w[1*n + c_off_c + 1] = 1
        
        U[0][col] = u[0]; U[1][col] = u[1]; U[2][col] = u[2]; U[3][col] = u[3]
        U[4][col] = u[4]; U[5][col] = u[5]; U[6][col] = u[6]; U[7][col] = u[7]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # M2 = (A21+A22)B11
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off+2] = 1; u[a_off+3] = 1
        v[b_off_r*m + b_off_c] = 1
        w[1*n + c_off_c] = 1
        w[1*n + c_off_c + 1] = 1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # M3 = A11(B12-B22)
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off] = 1
        v[b_off_r*m + b_off_c + 1] = 1
        v[(b_off_r+1)*m + (b_off_c+1)] = -1
        w[0*n + c_off_c + 1] = 1
        w[1*n + c_off_c + 1] = 1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # M4 = A22(B21-B11)
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off+3] = 1
        v[(b_off_r+1)*m + b_off_c] = 1
        v[b_off_r*m + b_off_c] = -1
        w[1*n + c_off_c] = 1
        w[0*n + c_off_c] = -1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # M5 = (A11+A12)B22
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off] = 1; u[a_off+1] = 1
        v[(b_off_r+1)*m + (b_off_c+1)] = 1
        w[0*n + c_off_c + 1] = 1
        w[1*n + c_off_c + 1] = 1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off+2] = 1; u[a_off] = -1
        v[b_off_r*m + b_off_c] = 1
        v[b_off_r*m + b_off_c + 1] = 1
        w[1*n + c_off_c] = 1
        w[0*n + c_off_c] = -1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off+1] = 1; u[a_off+3] = -1
        v[(b_off_r+1)*m + b_off_c] = 1
        v[(b_off_r+1)*m + (b_off_c+1)] = 1
        w[0*n + c_off_c] = 1
        w[1*n + c_off_c] = 1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1

    # Add Strassen for C1 (A1 B11 + A2 B21)
    # A1: a_off=0. B11: b_off_r=0, b_off_c=0. C1: c_off_c=0.
    add_strassen_2x2(0, 0, 0, 0)
    # A2: a_off=2. B21: b_off_r=2, b_off_c=0. C1: c_off_c=0.
    add_strassen_2x2(2, 2, 0, 0)
    
    # Add Strassen for C2 (A1 B12 + A2 B22)
    # A1: a_off=0. B12: b_off_r=0, b_off_c=2. C2: c_off_c=2.
    add_strassen_2x2(0, 0, 2, 2)
    # A2: a_off=2. B22: b_off_r=2, b_off_c=2. C2: c_off_c=2.
    add_strassen_2x2(2, 2, 2, 2)
    
    # Add Schoolbook for C3 (A1 B13 + A2 B23)
    # B13 is 2x3. Rows 0,1. Cols 4,5,6.
    # B23 is 2x3. Rows 2,3. Cols 4,5,6.
    
    def add_schoolbook_2x3(a_off, b_off_r, b_off_c, c_off_c):
        nonlocal col
        # A is 2x2. B is 2x3. C is 2x3.
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
        # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
        # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
        # ...
        
        # Term 1: A[0,0]B[0,0]
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off] = 1
        v[b_off_r*m + b_off_c] = 1
        w[0*n + c_off_c] = 1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # Term 2: A[0,1]B[1,0]
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off+1] = 1
        v[(b_off_r+1)*m + b_off_c] = 1
        w[0*n + c_off_c] = 1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # Term 3: A[0,0]B[0,1]
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off] = 1
        v[b_off_r*m + b_off_c + 1] = 1
        w[0*n + c_off_c + 1] = 1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # Term 4: A[0,1]B[1,1]
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off+1] = 1
        v[(b_off_r+1)*m + b_off_c + 1] = 1
        w[0*n + c_off_c + 1] = 1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # Term 5: A[0,0]B[0,2]
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off] = 1
        v[b_off_r*m + b_off_c + 2] = 1
        w[0*n + c_off_c + 2] = 1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # Term 6: A[0,1]B[1,2]
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off+1] = 1
        v[(b_off_r+1)*m + b_off_c + 2] = 1
        w[0*n + c_off_c + 2] = 1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # Term 7: A[1,0]B[0,0]
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off+2] = 1
        v[b_off_r*m + b_off_c] = 1
        w[1*n + c_off_c] = 1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # Term 8: A[1,1]B[1,0]
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off+3] = 1
        v[(b_off_r+1)*m + b_off_c] = 1
        w[1*n + c_off_c] = 1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # Term 9: A[1,0]B[0,1]
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off+2] = 1
        v[b_off_r*m + b_off_c + 1] = 1
        w[1*n + c_off_c + 1] = 1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # Term 10: A[1,1]B[1,1]
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off+3] = 1
        v[(b_off_r+1)*m + b_off_c + 1] = 1
        w[1*n + c_off_c + 1] = 1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # Term 11: A[1,0]B[0,2]
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off+2] = 1
        v[b_off_r*m + b_off_c + 2] = 1
        w[1*n + c_off_c + 2] = 1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1
        
        # Term 12: A[1,1]B[1,2]
        u = [0]*8; v = [0]*28; w = [0]*14
        u[a_off+3] = 1
        v[(b_off_r+1)*m + b_off_c + 2] = 1
        w[1*n + c_off_c + 2] = 1
        for i in range(8): U[i][col] = u[i]
        for i in range(28): V[i][col] = v[i]
        for i in range(14): W[i][col] = w[i]
        col += 1

    # Add Schoolbook for A1 B13
    add_schoolbook_2x3(0, 0, 4, 4)
    # Add Schoolbook for A2 B23
    add_schoolbook_2x3(2, 2, 4, 4)
    
    return {"U": U, "V": V, "W": W}
