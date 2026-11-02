def construct():
    n, m, p = 4, 4, 8
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        """Add a rank term. u is dict {flat_idx: coeff}, v is dict, w is dict."""
        u_vec = [0] * (n * m)
        for idx, coeff in u.items():
            u_vec[idx] += coeff
        v_vec = [0] * (m * p)
        for idx, coeff in v.items():
            v_vec[idx] += coeff
        w_vec = [0] * (p * n)
        for idx, coeff in w.items():
            w_vec[idx] += coeff
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # For 4x4 x 4x4, define block indices
    # A[i,j] flat index: i*4+j
    # B[j,k] flat index: j*4+k
    # C[i,k] flat index (trace): k*4+i
    
    def a_idx(i, j):
        return i * 4 + j
    
    def b_idx(j, k):
        return j * 4 + k
    
    def c_idx(i, k):
        return k * 4 + i
    
    # Strassen for 4x4 x 4x4
    # Split into 2x2 blocks
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Block A00: A[0,0], A[0,1], A[1,0], A[1,1]
    # Block A01: A[0,2], A[0,3], A[1,2], A[1,3]
    # Block A10: A[2,0], A[2,1], A[3,0], A[3,1]
    # Block A11: A[2,2], A[2,3], A[3,2], A[3,3]
    
    # Similarly for B and C
    
    # Strassen products (each is a 2x2 x 2x2 product):
    # P0 = (A00+A11) * (B00+B11)
    # P1 = (A10+A11) * B00
    # P2 = A01 * (B10-B00)
    # P3 = A11 * (B11-B01)
    # P4 = (A00+A01) * B11
    # P5 = (A10-A00) * (B00+B01)
    # P6 = (A01-A11) * (B10+B11)
    
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    # For each Pi, I do a 2x2 schoolbook multiplication (8 scalar products)
    
    # Define the 2x2 block entries
    # Block (bi, bj) of A: entries A[2*bi+r, 2*bj+c] for r,c in 0,1
    def a_block(bi, bj, r, c):
        return a_idx(2*bi+r, 2*bj+c)
    
    def b_block(bi, bj, r, c):
        return b_idx(2*bi+r, 2*bj+c)
    
    def c_block(bi, bj, r, c):
        return c_idx(2*bi+r, 2*bj+c)
    
    # For a 2x2 product X*Y, where X and Y are 2x2 matrices:
    # Result[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
    # Result[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
    # Result[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
    # Result[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
    
    # Each scalar product contributes to the result with a specific sign
    
    # I'll define each Strassen product as:
    # (A_entries, B_entries, C_contributions)
    # where A_entries is a list of 4 (flat_idx, sign) for the 2x2 A matrix
    # B_entries is a list of 4 (flat_idx, sign) for the 2x2 B matrix
    # C_contributions is a dict mapping (result_row, result_col) to (c_flat_idx, sign)
    
    # Actually, let me just enumerate all 8 scalar products for each Strassen product
    
    # P0 = (A00+A11) * (B00+B11)
    # A matrix entries:
    #   [0,0]: A[0,0]+A[2,2] -> {a_idx(0,0):1, a_idx(2,2):1}
    #   [0,1]: A[0,1]+A[2,3] -> {a_idx(0,1):1, a_idx(2,3):1}
    #   [1,0]: A[1,0]+A[3,2] -> {a_idx(1,0):1, a_idx(3,2):1}
    #   [1,1]: A[1,1]+A[3,3] -> {a_idx(1,1):1, a_idx(3,3):1}
    # B matrix entries:
    #   [0,0]: B[0,0]+B[2,2] -> {b_idx(0,0):1, b_idx(2,2):1}
    #   [0,1]: B[0,1]+B[2,3] -> {b_idx(0,1):1, b_idx(2,3):1}
    #   [1,0]: B[1,0]+B[3,2] -> {b_idx(1,0):1, b_idx(3,2):1}
    #   [1,1]: B[1,1]+B[3,3] -> {b_idx(1,1):1, b_idx(3,3):1}
    # C contributions (P0 contributes to C00 with +1, C11 with +1):
    #   Result[0,0] -> C[0,0] with +1, C[2,2] with +1
    #   Result[0,1] -> C[0,1] with +1, C[2,3] with +1
    #   Result[1,0] -> C[1,0] with +1, C[3,2] with +1
    #   Result[1,1] -> C[1,1] with +1, C[3,3] with +1
    
    # Wait, I need to be more careful. P0 is a 2x2 matrix that contributes to C00 (top-left 2x2 block of C) and C11 (bottom-right 2x2 block of C).
    
    # C00 = P0 + P3 - P4 + P6 (all 2x2 matrices)
    # So P0's [r,c] entry contributes to C[2*0+r, 2*0+c] = C[r,c] with sign +1
    # And P0's [r,c] entry contributes to C[2*1+r, 2*1+c] = C[2+r,2+c] with sign +1
    
    # Let me define a function to add a 2x2 product
    
    def add_2x2_product(A_entries, B_entries, C_contribs):
        """
        A_entries: list of 4 dicts, A_entries[r*2+c] is the dict for A[r,c]
        B_entries: list of 4 dicts, B_entries[r*2+c] is the dict for B[r,c]
        C_contribs: dict mapping (r,c) to list of (c_flat_idx, sign)
        """
        for r in range(2):
            for c in range(2):
                # C_result[r,c] = sum_k A[r,k]*B[k,c]
                for k in range(2):
                    # Scalar product: A[r,k] * B[k,c]
                    u = A_entries[r*2+k]
                    v = B_entries[k*2+c]
                    w = {}
                    for (ci, sign) in C_contribs[(r,c)]:
                        w[ci] = w.get(ci, 0) + sign
                    add_term(u, v, w)
    
    # Now define the 7 Strassen products
    
    # P0 = (A00+A11) * (B00+B11)
    A_P0 = [
        {a_idx(0,0):1, a_idx(2,2):1},  # [0,0]
        {a_idx(0,1):1, a_idx(2,3):1},  # [0,1]
        {a_idx(1,0):1, a_idx(3,2):1},  # [1,0]
        {a_idx(1,1):1, a_idx(3,3):1},  # [1,1]
    ]
    B_P0 = [
        {b_idx(0,0):1, b_idx(2,2):1},  # [0,0]
        {b_idx(0,1):1, b_idx(2,3):1},  # [0,1]
        {b_idx(1,0):1, b_idx(3,2):1},  # [1,0]
        {b_idx(1,1):1, b_idx(3,3):1},  # [1,1]
    ]
    C_P0 = {
        (0,0): [(c_idx(0,0),1), (c_idx(2,2),1)],
        (0,1): [(c_idx(0,1),1), (c_idx(2,3),1)],
        (1,0): [(c_idx(1,0),1), (c_idx(3,2),1)],
        (1,1): [(c_idx(1,1),1), (c_idx(3,3),1)],
    }
    add_2x2_product(A_P0, B_P0, C_P0)
    
    # P1 = (A10+A11) * B00
    A_P1 = [
        {a_idx(2,0):1, a_idx(2,2):1},  # [0,0]
        {a_idx(2,1):1, a_idx(2,3):1},  # [0,1]
        {a_idx(3,0):1, a_idx(3,2):1},  # [1,0]
        {a_idx(3,1):1, a_idx(3,3):1},  # [1,1]
    ]
    B_P1 = [
        {b_idx(0,0):1},  # [0,0]
        {b_idx(0,1):1},  # [0,1]
        {b_idx(1,0):1},  # [1,0]
        {b_idx(1,1):1},  # [1,1]
    ]
    C_P1 = {
        (0,0): [(c_idx(2,0),1)],
        (0,1): [(c_idx(2,1),1)],
        (1,0): [(c_idx(3,0),1)],
        (1,1): [(c_idx(3,1),1)],
    }
    add_2x2_product(A_P1, B_P1, C_P1)
    
    # P2 = A01 * (B10-B00)
    A_P2 = [
        {a_idx(0,2):1},  # [0,0]
        {a_idx(0,3):1},  # [0,1]
        {a_idx(1,2):1},  # [1,0]
        {a_idx(1,3):1},  # [1,1]
    ]
    B_P2 = [
        {b_idx(2,0):1, b_idx(0,0):-1},  # [0,0]
        {b_idx(2,1):1, b_idx(0,1):-1},  # [0,1]
        {b_idx(3,0):1, b_idx(1,0):-1},  # [1,0]
        {b_idx(3,1):1, b_idx(1,1):-1},  # [1,1]
    ]
    C_P2 = {
        (0,0): [(c_idx(0,1),1), (c_idx(2,3),1)],
        (0,1): [(c_idx(0,2),1), (c_idx(2,0),1)],
        (1,0): [(c_idx(1,1),1), (c_idx(3,3),1)],
        (1,1): [(c_idx(1,2),1), (c_idx(3,0),1)],
    }
    add_2x2_product(A_P2, B_P2, C_P2)
    
    # P3 = A11 * (B11-B01)
    A_P3 = [
        {a_idx(2,2):1},  # [0,0]
        {a_idx(2,3):1},  # [0,1]
        {a_idx(3,2):1},  # [1,0]
        {a_idx(3,3):1},  # [1,1]
    ]
    B_P3 = [
        {b_idx(2,2):1, b_idx(0,2):-1},  # [0,0]
        {b_idx(2,3):1, b_idx(0,3):-1},  # [0,1]
        {b_idx(3,2):1, b_idx(1,2):-1},  # [1,0]
        {b_idx(3,3):1, b_idx(1,3):-1},  # [1,1]
    ]
    C_P3 = {
        (0,0): [(c_idx(0,0),1), (c_idx(2,2),1)],
        (0,1): [(c_idx(0,1),1), (c_idx(2,3),1)],
        (1,0): [(c_idx(1,0),1), (c_idx(3,2),1)],
        (1,1): [(c_idx(1,1),1), (c_idx(3,3),1)],
    }
    add_2x2_product(A_P3, B_P3, C_P3)
    
    # P4 = (A00+A01) * B11
    A_P4 = [
        {a_idx(0,0):1, a_idx(0,2):1},  # [0,0]
        {a_idx(0,1):1, a_idx(0,3):1},  # [0,1]
        {a_idx(1,0):1, a_idx(1,2):1},  # [1,0]
        {a_idx(1,1):1, a_idx(1,3):1},  # [1,1]
    ]
    B_P4 = [
        {b_idx(2,2):1},  # [0,0]
        {b_idx(2,3):1},  # [0,1]
        {b_idx(3,2):1},  # [1,0]
        {b_idx(3,3):1},  # [1,1]
    ]
    C_P4 = {
        (0,0): [(c_idx(0,0),-1), (c_idx(2,2),-1)],
        (0,1): [(c_idx(0,1),1), (c_idx(2,3),1)],
        (1,0): [(c_idx(1,0),-1), (c_idx(3,2),-1)],
        (1,1): [(c_idx(1,1),1), (c_idx(3,3),1)],
    }
    add_2x2_product(A_P4, B_P4, C_P4)
    
    # P5 = (A10-A00) * (B00+B01)
    A_P5 = [
        {a_idx(2,0):1, a_idx(0,0):-1},  # [0,0]
        {a_idx(2,1):1, a_idx(0,1):-1},  # [0,1]
        {a_idx(3,0):1, a_idx(1,0):-1},  # [1,0]
        {a_idx(3,1):1, a_idx(1,1):-1},  # [1,1]
    ]
    B_P5 = [
        {b_idx(0,0):1, b_idx(0,2):1},  # [0,0]
        {b_idx(0,1):1, b_idx(0,3):1},  # [0,1]
        {b_idx(1,0):1, b_idx(1,2):1},  # [1,0]
        {b_idx(1,1):1, b_idx(1,3):1},  # [1,1]
    ]
    C_P5 = {
        (0,0): [(c_idx(2,3),1)],
        (0,1): [(c_idx(2,0),1)],
        (1,0): [(c_idx(3,3),1)],
        (1,1): [(c_idx(3,0),1)],
    }
    add_2x2_product(A_P5, B_P5, C_P5)
    
    # P6 = (A01-A11) * (B10+B11)
    A_P6 = [
        {a_idx(0,2):1, a_idx(2,2):-1},  # [0,0]
        {a_idx(0,3):1, a_idx(2,3):-1},  # [0,1]
        {a_idx(1,2):1, a_idx(3,2):-1},  # [1,0]
        {a_idx(1,3):1, a_idx(3,3):-1},  # [1,1]
    ]
    B_P6 = [
        {b_idx(2,0):1, b_idx(2,2):1},  # [0,0]
        {b_idx(2,1):1, b_idx(2,3):1},  # [0,1]
        {b_idx(3,0):1, b_idx(3,2):1},  # [1,0]
        {b_idx(3,1):1, b_idx(3,3):1},  # [1,1]
    ]
    C_P6 = {
        (0,0): [(c_idx(0,0),1)],
        (0,1): [(c_idx(0,1),1)],
        (1,0): [(c_idx(1,0),1)],
        (1,1): [(c_idx(1,1),1)],
    }
    add_2x2_product(A_P6, B_P6, C_P6)
    
    # That's 7 * 8 = 56 terms for one 4x4 x 4x4 multiplication
    # Now I need to do this twice for the 4x4 x 4x8 case
    
    # Wait, I already added 56 terms. I need to add another 56 for the second half of B.
    # The second half uses B[:, 4:8] instead of B[:, 0:4]
    
    # Let me redo this. I'll build the decomposition for B[:, 0:4] and B[:, 4:8] separately.
    
    # Actually, I already built it for B[:, 0:4] (using b_idx which maps to j*4+k for j,k in 0..3).
    # For B[:, 4:8], I need to use b_idx(j, k+4) which is j*4+(k+4) = j*4+k+4
    
    # Let me clear and rebuild properly.
    
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()
    
    # Build for B[:, 0:4] (k in 0..3)
    # Then build for B[:, 4:8] (k in 4..7)
    
    # I'll parameterize the b_idx function
    
    def build_strassen(k_offset):
        def b_idx2(j, k):
            return j * 4 + (k + k_offset)
        
        # P0 = (A00+A11) * (B00+B11)
        A_P0 = [
            {a_idx(0,0):1, a_idx(2,2):1},
            {a_idx(0,1):1, a_idx(2,3):1},
            {a_idx(1,0):1, a_idx(3,2):1},
            {a_idx(1,1):1, a_idx(3,3):1},
        ]
        B_P0 = [
            {b_idx2(0,0):1, b_idx2(2,2):1},
            {b_idx2(0,1):1, b_idx2(2,3):1},
            {b_idx2(1,0):1, b_idx2(3,2):1},
            {b_idx2(1,1):1, b_idx2(3,3):1},
        ]
        C_P0 = {
            (0,0): [(c_idx(0,0),1), (c_idx(2,2),1)],
            (0,1): [(c_idx(0,1),1), (c_idx(2,3),1)],
            (1,0): [(c_idx(1,0),1), (c_idx(3,2),1)],
            (1,1): [(c_idx(1,1),1), (c_idx(3,3),1)],
        }
        add_2x2_product(A_P0, B_P0, C_P0)
        
        # P1 = (A10+A11) * B00
        A_P1 = [
            {a_idx(2,0):1, a_idx(2,2):1},
            {a_idx(2,1):1, a_idx(2,3):1},
            {a_idx(3,0):1, a_idx(3,2):1},
            {a_idx(3,1):1, a_idx(3,3):1},
        ]
        B_P1 = [
            {b_idx2(0,0):1},
            {b_idx2(0,1):1},
            {b_idx2(1,0):1},
            {b_idx2(1,1):1},
        ]
        C_P1 = {
            (0,0): [(c_idx(2,0),1)],
            (0,1): [(c_idx(2,1),1)],
            (1,0): [(c_idx(3,0),1)],
            (1,1): [(c_idx(3,1),1)],
        }
        add_2x2_product(A_P1, B_P1, C_P1)
        
        # P2 = A01 * (B10-B00)
        A_P2 = [
            {a_idx(0,2):1},
            {a_idx(0,3):1},
            {a_idx(1,2):1},
            {a_idx(1,3):1},
        ]
        B_P2 = [
            {b_idx2(2,0):1, b_idx2(0,0):-1},
            {b_idx2(2,1):1, b_idx2(0,1):-1},
            {b_idx2(3,0):1, b_idx2(1,0):-1},
            {b_idx2(3,1):1, b_idx2(1,1):-1},
        ]
        C_P2 = {
            (0,0): [(c_idx(0,1),1), (c_idx(2,3),1)],
            (0,1): [(c_idx(0,2),1), (c_idx(2,0),1)],
            (1,0): [(c_idx(1,1),1), (c_idx(3,3),1)],
            (1,1): [(c_idx(1,2),1), (c_idx(3,0),1)],
        }
        add_2x2_product(A_P2, B_P2, C_P2)
        
        # P3 = A11 * (B11-B01)
        A_P3 = [
            {a_idx(2,2):1},
            {a_idx(2,3):1},
            {a_idx(3,2):1},
            {a_idx(3,3):1},
        ]
        B_P3 = [
            {b_idx2(2,2):1, b_idx2(0,2):-1},
            {b_idx2(2,3):1, b_idx2(0,3):-1},
            {b_idx2(3,2):1, b_idx2(1,2):-1},
            {b_idx2(3,3):1, b_idx2(1,3):-1},
        ]
        C_P3 = {
            (0,0): [(c_idx(0,0),1), (c_idx(2,2),1)],
            (0,1): [(c_idx(0,1),1), (c_idx(2,3),1)],
            (1,0): [(c_idx(1,0),1), (c_idx(3,2),1)],
            (1,1): [(c_idx(1,1),1), (c_idx(3,3),1)],
        }
        add_2x2_product(A_P3, B_P3, C_P3)
        
        # P4 = (A00+A01) * B11
        A_P4 = [
            {a_idx(0,0):1, a_idx(0,2):1},
            {a_idx(0,1):1, a_idx(0,3):1},
            {a_idx(1,0):1, a_idx(1,2):1},
            {a_idx(1,1):1, a_idx(1,3):1},
        ]
        B_P4 = [
            {b_idx2(2,2):1},
            {b_idx2(2,3):1},
            {b_idx2(3,2):1},
            {b_idx2(3,3):1},
        ]
        C_P4 = {
            (0,0): [(c_idx(0,0),-1), (c_idx(2,2),-1)],
            (0,1): [(c_idx(0,1),1), (c_idx(2,3),1)],
            (1,0): [(c_idx(1,0),-1), (c_idx(3,2),-1)],
            (1,1): [(c_idx(1,1),1), (c_idx(3,3),1)],
        }
        add_2x2_product(A_P4, B_P4, C_P4)
        
        # P5 = (A10-A00) * (B00+B01)
        A_P5 = [
            {a_idx(2,0):1, a_idx(0,0):-1},
            {a_idx(2,1):1, a_idx(0,1):-1},
            {a_idx(3,0):1, a_idx(1,0):-1},
            {a_idx(3,1):1, a_idx(1,1):-1},
        ]
        B_P5 = [
            {b_idx2(0,0):1, b_idx2(0,2):1},
            {b_idx2(0,1):1, b_idx2(0,3):1},
            {b_idx2(1,0):1, b_idx2(1,2):1},
            {b_idx2(1,1):1, b_idx2(1,3):1},
        ]
        C_P5 = {
            (0,0): [(c_idx(2,3),1)],
            (0,1): [(c_idx(2,0),1)],
            (1,0): [(c_idx(3,3),1)],
            (1,1): [(c_idx(3,0),1)],
        }
        add_2x2_product(A_P5, B_P5, C_P5)
        
        # P6 = (A01-A11) * (B10+B11)
        A_P6 = [
            {a_idx(0,2):1, a_idx(2,2):-1},
            {a_idx(0,3):1, a_idx(2,3):-1},
            {a_idx(1,2):1, a_idx(3,2):-1},
            {a_idx(1,3):1, a_idx(3,3):-1},
        ]
        B_P6 = [
            {b_idx2(2,0):1, b_idx2(2,2):1},
            {b_idx2(2,1):1, b_idx2(2,3):1},
            {b_idx2(3,0):1, b_idx2(3,2):1},
            {b_idx2(3,1):1, b_idx2(3,3):1},
        ]
        C_P6 = {
            (0,0): [(c_idx(0,0),1)],
            (0,1): [(c_idx(0,1),1)],
            (1,0): [(c_idx(1,0),1)],
            (1,1): [(c_idx(1,1),1)],
        }
        add_2x2_product(A_P6, B_P6, C_P6)
    
    build_strassen(0)  # B[:, 0:4]
    build_strassen(4)  # B[:, 4:8]
    
    rank = len(U_cols)
    U = [row for row in U_cols]
    V = [row for row in V_cols]
    W = [row for row in W_cols]
    
    return {"U": U, "V": V, "W": W}
