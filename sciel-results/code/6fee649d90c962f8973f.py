import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Helper to add a term
    # u_vec: length 16 (A)
    # v_vec: length 32 (B)
    # w_vec: length 32 (C)
    # col: index in U, V, W
    
    # We need to define the Strassen decomposition logic.
    
    # 1. Define 2x2 Strassen scalar products
    # Inputs: x11, x12, x21, x22, y11, y12, y21, y22
    # Outputs: 7 products P1..P7
    # Each P is (u_coeff, v_coeff)
    # u_coeff is for X, v_coeff is for Y
    
    def strassen_2x2_products():
        # Returns list of 7 tuples (u_vec, v_vec)
        # u_vec is for X entries, v_vec for Y entries
        # X entries indices: 0:x11, 1:x12, 2:x21, 3:x22
        # Y entries indices: 0:y11, 1:y12, 2:y21, 3:y22
        
        prods = []
        # P1: (x11+x22)(y11+y22)
        u = np.array([1, 0, 0, 1])
        v = np.array([1, 0, 0, 1])
        prods.append((u, v))
        
        # P2: (x21+x22)y11
        u = np.array([0, 0, 1, 1])
        v = np.array([1, 0, 0, 0])
        prods.append((u, v))
        
        # P3: x11(y12-y22)
        u = np.array([1, 0, 0, 0])
        v = np.array([0, 1, 0, -1])
        prods.append((u, v))
        
        # P4: x22(y21-y11)
        u = np.array([0, 0, 0, 1])
        v = np.array([-1, 0, 1, 0])
        prods.append((u, v))
        
        # P5: (x11+x12)y22
        u = np.array([1, 1, 0, 0])
        v = np.array([0, 0, 0, 1])
        prods.append((u, v))
        
        # P6: (x21-x11)(y11+y12)
        u = np.array([-1, 0, 1, 0])
        v = np.array([1, 1, 0, 0])
        prods.append((u, v))
        
        # P7: (x12-x22)(y21+y22)
        u = np.array([0, 1, 0, -1])
        v = np.array([0, 0, 1, 1])
        prods.append((u, v))
        
        return prods

    # 2. Define 4x4 Strassen block products
    # We need to map the 2x2 Strassen inputs (X, Y) to the 4x4 blocks (A, B_s)
    
    # A blocks:
    # A11: 0-1, 0-1
    # A12: 0-1, 2-3
    # A21: 2-3, 0-1
    # A22: 2-3, 2-3
    
    # B_s blocks:
    # B11: 0-1, 0-1 (cols 0-1)
    # B12: 0-1, 2-3 (cols 2-3)
    # B21: 2-3, 0-1 (cols 0-1)
    # B22: 2-3, 2-3 (cols 2-3)
    
    # We will generate the 7 products for a given s (0 or 1)
    # s=0 corresponds to B cols 0-3
    # s=1 corresponds to B cols 4-7
    
    def get_4x4_strassen_terms(s):
        # s: 0 or 1
        # Returns list of 7 * 7 = 49 terms
        # Each term is (u_A, v_B, w_C)
        # But w_C is not yet determined, we need to trace back from C.
        # Actually, we can determine u_A and v_B for each scalar product.
        # Then we determine the coefficient in C for each scalar product.
        
        terms = []
        strassen_2x2 = strassen_2x2_products()
        
        # Define the 7 block products for 4x4
        # Each is (X_block, Y_block)
        # X_block is a 2x2 matrix of linear forms in A
        # Y_block is a 2x2 matrix of linear forms in B_s
        
        # We represent a 2x2 matrix of linear forms as a list of 4 vectors
        # vec[0] for 11, vec[1] for 12, vec[2] for 21, vec[3] for 22
        
        # A vectors (length 16)
        # A11: indices 0,1,4,5? No.
        # A is 4x4. Index i*m+j.
        # A11: i=0,1; j=0,1 -> 0,1,4,5
        # A12: i=0,1; j=2,3 -> 2,3,6,7
        # A21: i=2,3; j=0,1 -> 8,9,12,13
        # A22: i=2,3; j=2,3 -> 10,11,14,15
        
        def A_block(r1, r2, c1, c2):
            # r1,r2: row indices, c1,c2: col indices
            vec = np.zeros(16)
            for i in range(r1, r2+1):
                for j in range(c1, c2+1):
                    vec[i*4+j] = 1
            return vec

        A11 = A_block(0,1,0,1)
        A12 = A_block(0,1,2,3)
        A21 = A_block(2,3,0,1)
        A22 = A_block(2,3,2,3)
        
        # B vectors (length 32)
        # B_s is 4x4. Index j*8 + k? No, V is m*p = 4*8=32.
        # Index j*p + k.
        # For s=0, k is 0..3.
        # For s=1, k is 4..7.
        
        def B_block(s, r1, r2, c1, c2):
            # r1,r2: row indices (j), c1,c2: col indices (k relative to s)
            vec = np.zeros(32)
            for j in range(r1, r2+1):
                for k in range(c1, c2+1):
                    k_abs = k + s*4
                    vec[j*8 + k_abs] = 1
            return vec

        B11 = B_block(s, 0,1,0,1)
        B12 = B_block(s, 0,1,2,3)
        B21 = B_block(s, 2,3,0,1)
        B22 = B_block(s, 2,3,2,3)
        
        # The 7 products
        # 1. (A11+A22) (B11+B22)
        X1 = [A11+A22, np.zeros(16), np.zeros(16), np.zeros(16)] # This is wrong. X is 2x2.
        # Strassen 4x4 products are of the form X * Y where X, Y are 2x2 matrices.
        # But in the formula (A11+A22)(B11+B22), X is a scalar? No.
        # A11+A22 is a 2x2 matrix. B11+B22 is a 2x2 matrix.
        # So X = A11+A22 (2x2 matrix of linear forms).
        # Y = B11+B22 (2x2 matrix of linear forms).
        
        # We need to represent X as a 2x2 matrix of vectors.
        # X[0,0] = A11+A22? No.
        # A11 is a 2x2 matrix. A22 is a 2x2 matrix.
        # So A11+A22 is a 2x2 matrix.
        # Its entries are linear forms in A.
        # X1[0,0] = A11[0,0] + A22[0,0]
        # X1[0,1] = A11[0,1] + A22[0,1]
        # etc.
        
        # Let's create a function to add 2x2 matrices of vectors.
        def add_2x2(M1, M2):
            return [[M1[i][j] + M2[i][j] for j in range(2)] for i in range(2)]
        
        def sub_2x2(M1, M2):
            return [[M1[i][j] - M2[i][j] for j in range(2)] for i in range(2)]
        
        # Represent A blocks as 2x2 matrices of vectors
        # A11 is a 2x2 matrix. Each entry is a vector (length 16) with 1 at the corresponding position.
        # Actually, A11 is the top-left 2x2 block of A.
        # A11[0,0] is A[0,0]. Vector e_0.
        # A11[0,1] is A[0,1]. Vector e_1.
        # A11[1,0] is A[1,0]. Vector e_4.
        # A11[1,1] is A[1,1]. Vector e_5.
        
        def make_A_matrix(r1, c1):
            # r1, c1: top-left corner of 2x2 block
            M = []
            for i in range(2):
                row = []
                for j in range(2):
                    vec = np.zeros(16)
                    vec[(r1+i)*4 + (c1+j)] = 1
                    row.append(vec)
                M.append(row)
            return M
            
        def make_B_matrix(s, r1, c1):
            M = []
            for i in range(2):
                row = []
                for j in range(2):
                    vec = np.zeros(32)
                    j_abs = r1+i
                    k_abs = c1+j + s*4
                    vec[j_abs*8 + k_abs] = 1
                    row.append(vec)
                M.append(row)
            return M
            
        A11_M = make_A_matrix(0,0)
        A12_M = make_A_matrix(0,2)
        A21_M = make_A_matrix(2,0)
        A22_M = make_A_matrix(2,2)
        
        B11_M = make_B_matrix(s, 0,0)
        B12_M = make_B_matrix(s, 0,2)
        B21_M = make_B_matrix(s, 2,0)
        B22_M = make_B_matrix(s, 2,2)
        
        # Define the 7 products
        prods_4x4 = [
            (add_2x2(A11_M, A22_M), add_2x2(B11_M, B22_M)), # 1
            (add_2x2(A21_M, A22_M), B11_M),                  # 2
            (A11_M, sub_2x2(B12_M, B22_M)),                  # 3
            (A22_M, sub_2x2(B21_M, B11_M)),                  # 4
            (add_2x2(A11_M, A12_M), B22_M),                  # 5
            (sub_2x2(A21_M, A11_M), add_2x2(B11_M, B12_M)),  # 6
            (sub_2x2(A12_M, A22_M), add_2x2(B21_M, B22_M))   # 7
        ]
        
        # For each product, apply 2x2 Strassen
        for q in range(7):
            X_M, Y_M = prods_4x4[q]
            # X_M is 2x2 matrix of vectors (length 16)
            # Y_M is 2x2 matrix of vectors (length 32)
            
            # Get the 7 scalar products
            # We need to map X_M entries to x11, x12, x21, x22
            # X_M[0][0] is x11, X_M[0][1] is x12, X_M[1][0] is x21, X_M[1][1] is x22
            
            x_vecs = [X_M[0][0], X_M[0][1], X_M[1][0], X_M[1][1]]
            y_vecs = [Y_M[0][0], Y_M[0][1], Y_M[1][0], Y_M[1][1]]
            
            # strassen_2x2_products returns (u, v) where u is for x, v is for y
            for t in range(7):
                u_x, v_y = strassen_2x2_products()[t]
                # u_x is length 4, v_y is length 4
                # Combine with x_vecs and y_vecs
                u_A = np.zeros(16)
                for idx in range(4):
                    u_A += u_x[idx] * x_vecs[idx]
                
                v_B = np.zeros(32)
                for idx in range(4):
                    v_B += v_y[idx] * y_vecs[idx]
                
                # Now we have u_A and v_B for this scalar product.
                # We need to determine the coefficient in C.
                # This scalar product contributes to Z_q (the q-th 4x4 Strassen product).
                # Z_q is a 2x2 matrix.
                # The contribution of P_t to Z_q[0,0], Z_q[0,1], etc. is given by the 2x2 Strassen result formulas.
                
                # Result formulas:
                # Z11 = P1 + P4 - P5 + P7
                # Z12 = P3 + P5
                # Z21 = P2 + P4
                # Z22 = P1 - P2 + P3 + P6
                
                coeff_z = [0, 0, 0, 0] # z11, z12, z21, z22
                if t == 0: # P1
                    coeff_z[0] += 1
                    coeff_z[3] += 1
                elif t == 1: # P2
                    coeff_z[2] += 1
                    coeff_z[3] -= 1
                elif t == 2: # P3
                    coeff_z[1] += 1
                    coeff_z[3] += 1
                elif t == 3: # P4
                    coeff_z[0] += 1
                    coeff_z[2] += 1
                elif t == 4: # P5
                    coeff_z[0] -= 1
                    coeff_z[1] += 1
                elif t == 5: # P6
                    coeff_z[3] += 1
                elif t == 6: # P7
                    coeff_z[0] += 1
                
                # Now Z_q contributes to C_s.
                # C_s11 = M1 + M4 - M5 + M7
                # C_s12 = M3 + M5
                # C_s21 = M2 + M4
                # C_s22 = M1 - M2 + M3 + M6
                
                coeff_c = [0, 0, 0, 0] # c11, c12, c21, c22 (relative to s)
                if q == 0: # M1
                    coeff_c[0] += 1
                    coeff_c[3] += 1
                elif q == 1: # M2
                    coeff_c[2] += 1
                    coeff_c[3] -= 1
                elif q == 2: # M3
                    coeff_c[1] += 1
                    coeff_c[3] += 1
                elif q == 3: # M4
                    coeff_c[0] += 1
                    coeff_c[2] += 1
                elif q == 4: # M5
                    coeff_c[0] -= 1
                    coeff_c[1] += 1
                elif q == 5: # M6
                    coeff_c[3] += 1
                elif q == 6: # M7
                    coeff_c[0] += 1
                
                # Combine coeff_z and coeff_c
                # C_s[i,j] = sum_q coeff_c[q][i,j] * Z_q[i,j]
                # Z_q[i,j] = sum_t coeff_z[t][i,j] * P_t
                # So P_t contributes coeff_c[q][i,j] * coeff_z[t][i,j] to C_s[i,j]
                
                w_C = np.zeros(32)
                for i in range(2):
                    for j in range(2):
                        val = coeff_c[i*2+j] * coeff_z[i*2+j]
                        if val != 0:
                            # C_s[i,j] corresponds to C row i, col j+s*2?
                            # C is 4x8.
                            # C_s is the part of C corresponding to B_s.
                            # B_s cols are 0-3 or 4-7.
                            # C_s cols are 0-3 or 4-7.
                            # C_s[i,j] is C[i, j + s*4]?
                            # Wait, B_s is 4x4. C_s is 4x4.
                            # C_s[i,j] corresponds to C[i, j + s*4].
                            row = i
                            col = j + s*4
                            # W index is k*n + i. Here k is col, i is row.
                            # W[col*4 + row]
                            w_C[col*4 + row] = val
                
                terms.append((u_A, v_B, w_C))
                
        return terms

    all_terms = []
    all_terms.extend(get_4x4_strassen_terms(0))
    all_terms.extend(get_4x4_strassen_terms(1))
    
    # Fill U, V, W
    for r, (u, v, w) in enumerate(all_terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
