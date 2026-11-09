import numpy as np

def construct():
    n, m, p = 2, 3, 5
    # Total rank 28
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx, col] = u_val
        V[v_idx, col] = v_val
        W[w_idx, col] = w_val
        col += 1

    # Helper for Strassen 2x2x2
    # A indices: (i, j) -> i*m + j
    # B indices: (j, k) -> j*p + k
    # C indices: (i, k) -> k*n + i  (Trace convention: W[k*n+i])
    def add_strassen(a_base, b_base, c_base):
        # a_base: (i, j) offset
        # b_base: (j, k) offset
        # c_base: (i, k) offset
        # Local indices:
        # A: 0,1,2,3 (11, 12, 21, 22)
        # B: 0,1,2,3 (11, 12, 21, 22)
        # C: 0,1,2,3 (11, 12, 21, 22)
        
        # M1 = (A11 + A22)(B11 + B22)
        add_term(a_base + 0, b_base + 0, c_base + 0, 1, 1, 1)
        add_term(a_base + 3, b_base + 0, c_base + 0, 1, 1, 1)
        add_term(a_base + 0, b_base + 1, c_base + 0, 1, 1, 1)
        add_term(a_base + 3, b_base + 1, c_base + 0, 1, 1, 1)
        add_term(a_base + 3, b_base + 2, c_base + 0, 1, 1, 1)
        add_term(a_base + 0, b_base + 3, c_base + 0, 1, 1, 1)
        # Wait, Strassen M1 is (A11+A22)(B11+B22).
        # This expands to A11B11 + A11B22 + A22B11 + A22B22.
        # But we are adding rank-1 terms.
        # (A11+A22) is not a single input.
        # We need to represent the bilinear form.
        # The term is (A11+A22) * (B11+B22).
        # This is NOT a rank-1 term in the original tensor space unless we change variables.
        # The tensor decomposition requires U, V, W to be linear combinations of inputs.
        # So U column can have 1s at A11 and A22.
        # V column can have 1s at B11 and B22.
        # W column contributes to C.
        
        # Let's re-implement add_strassen properly.
        pass

    # Correct Strassen Implementation
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
        # a_r, a_c: row/col of A block (2x2)
        # b_r, b_c: row/col of B block (2x2)
        # c_r, c_c: row/col of C block (2x2)
        
        # M1 = (A11 + A22)(B11 + B22)
        U[a_r*m + a_c, col] = 1
        U[(a_r+1)*m + (a_c+1), col] = 1
        V[b_r*p + b_c, col] = 1
        V[(b_r+1)*p + (b_c+1), col] = 1
        W[c_c*n + c_r, col] = 1
        W[(c_c+1)*n + (c_r+1), col] = 1
        W[c_c*n + (c_r+1), col] = 1 # C21? No.
        # C11 gets M1, C22 gets M1, C12 gets -M5+M3, C21 gets M4+M2.
        # Let's use the standard formula.
        
        # M1
        U[a_r*m + a_c, col] = 1
        U[(a_r+1)*m + (a_c+1), col] = 1
        V[b_r*p + b_c, col] = 1
        V[(b_r+1)*p + (b_c+1), col] = 1
        W[c_c*n + c_r, col] = 1      # C11
        W[(c_c+1)*n + (c_r+1), col] = 1 # C22
        col += 1

        # M2 = (A21 + A22)B11
        U[(a_r+1)*m + a_c, col] = 1
        U[(a_r+1)*m + (a_c+1), col] = 1
        V[b_r*p + b_c, col] = 1
        W[(c_c+1)*n + c_r, col] = 1 # C21
        col += 1

        # M3 = A11(B12 - B22)
        U[a_r*m + a_c, col] = 1
        V[b_r*p + (b_c+1), col] = 1
        V[(b_r+1)*p + (b_c+1), col] = -1
        W[c_c*n + (c_r+1), col] = 1 # C12
        col += 1

        # M4 = A22(B21 - B11)
        U[(a_r+1)*m + (a_c+1), col] = 1
        V[(b_r+1)*p + b_c, col] = 1
        V[b_r*p + b_c, col] = -1
        W[c_c*n + c_r, col] = 1 # C11
        W[(c_c+1)*n + c_r, col] = 1 # C21
        col += 1

        # M5 = (A11 + A12)B22
        U[a_r*m + a_c, col] = 1
        U[a_r*m + (a_c+1), col] = 1
        V[(b_r+1)*p + (b_c+1), col] = 1
        W[c_c*n + c_r, col] = -1 # C11
        W[c_c*n + (c_r+1), col] = 1 # C12
        col += 1

        # M6 = (A21 - A11)(B11 + B12)
        U[(a_r+1)*m + a_c, col] = 1
        U[a_r*m + a_c, col] = -1
        V[b_r*p + b_c, col] = 1
        V[b_r*p + (b_c+1), col] = 1
        W[(c_c+1)*n + (c_r+1), col] = 1 # C22
        col += 1

        # M7 = (A12 - A22)(B21 + B22)
        U[a_r*m + (a_c+1), col] = 1
        U[(a_r+1)*m + (a_c+1), col] = -1
        V[(b_r+1)*p + b_c, col] = 1
        V[(b_r+1)*p + (b_c+1), col] = 1
        W[c_c*n + c_r, col] = 1 # C11
        W[(c_c+1)*n + (c_r+1), col] = 1 # C22
        col += 1

    # Helper for 2x2x1
    def add_221_block(a_r, a_c, b_r, b_c, c_r, c_c):
        # A 2x2, B 2x1, C 2x1
        # C1 = A11B11 + A12B21
        # C2 = A21B11 + A22B21
        
        # T1 = A11 B11 -> C1
        U[a_r*m + a_c, col] = 1
        V[b_r*p + b_c, col] = 1
        W[c_c*n + c_r, col] = 1
        col += 1
        
        # T2 = A12 B21 -> C1
        U[a_r*m + (a_c+1), col] = 1
        V[(b_r+1)*p + b_c, col] = 1
        W[c_c*n + c_r, col] = 1
        col += 1
        
        # T3 = A21 B11 -> C2
        U[(a_r+1)*m + a_c, col] = 1
        V[b_r*p + b_c, col] = 1
        W[c_c*n + (c_r+1), col] = 1
        col += 1
        
        # T4 = A22 B21 -> C2
        U[(a_r+1)*m + (a_c+1), col] = 1
        V[(b_r+1)*p + b_c, col] = 1
        W[c_c*n + (c_r+1), col] = 1
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:3] -> C[0:2, 2:3]
    # B rows 0:2, col 2. C rows 0:2, col 2.
    add_221_block(0, 0, 0, 2, 0, 2)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 3:5] -> C[0:2, 3:5]
    add_strassen_block(0, 0, 0, 3, 0, 3)
    
    # Block 4: A[0:2, 2:3] * B[2:3, 0:5] -> C[0:2, 0:5]
    # A row 0:2, col 2. B row 2, col 0:5.
    # This is 2x1x5.
    # C_ik = A_i2 B_2k
    for k in range(5):
        # C_1k (i=0)
        U[0*m + 2, col] = 1 # A_13 (0-indexed: 0, 2)
        V[2*p + k, col] = 1 # B_3k (2-indexed: 2, k)
        W[k*n + 0, col] = 1 # C_1k
        col += 1
        # C_2k (i=1)
        U[1*m + 2, col] = 1 # A_23
        V[2*p + k, col] = 1 # B_3k
        W[k*n + 1, col] = 1 # C_2k
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
