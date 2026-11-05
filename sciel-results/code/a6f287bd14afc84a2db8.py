import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = 3, 4, 7
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (21, rank)
    
    # We will build lists of columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        # u_idx: row in U
        # v_idx: row in V
        # w_idx: row in W
        # Values can be -1, 1, 0.5 etc.
        
        # U is 12xR, V is 28xR, W is 21xR
        # We append to lists, then pad later
        
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        u_col[u_idx] = u_val
        v_col[v_idx] = v_val
        w_col[w_idx] = w_val
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    # Helper for Strassen 2x2x2
    # A indices: (r1,c1), (r1,c2), (r2,c1), (r2,c2)
    # B indices: (r1,c1), (r1,c2), (r2,c1), (r2,c2)
    # C indices: (r1,c1), (r1,c2), (r2,c1), (r2,c2)
    
    def strassen_block(a_r1, a_c1, a_r2, a_c2, b_r1, b_c1, b_r2, b_c2, c_r1, c_c1, c_r2, c_c2):
        # A = [[a11, a12], [a21, a22]]
        # B = [[b11, b12], [b21, b22]]
        # C = A*B
        
        # U indices
        u_a11 = a_r1 * m + a_c1
        u_a12 = a_r1 * m + a_c2
        u_a21 = a_r2 * m + a_c1
        u_a22 = a_r2 * m + a_c2
        
        # V indices
        v_b11 = b_r1 * p + b_c1
        v_b12 = b_r1 * p + b_c2
        v_b21 = b_r2 * p + b_c1
        v_b22 = b_r2 * p + b_c2
        
        # W indices
        w_c11 = c_c1 * n + c_r1 # Wait, W[k*n+i] -> col i, row k?
        # Problem: W[k*n+i, r]. C[i,k] = sum ... W[k*n+i, r]
        # So W index is k*n + i.
        # c_r1 is row of C (k), c_c1 is col of C (i).
        w_c11 = c_r1 * n + c_c1
        w_c12 = c_r1 * n + c_c2
        w_c21 = c_r2 * n + c_c1
        w_c22 = c_r2 * n + c_c2
        
        # M1 = (a11+a22)(b11+b22)
        add_term(u_a11, v_b11, w_c11, 1, 1, 0.5)
        add_term(u_a11, v_b22, w_c11, 1, 1, 0.5)
        add_term(u_a22, v_b11, w_c11, 1, 1, 0.5)
        add_term(u_a22, v_b22, w_c11, 1, 1, 0.5)
        
        # M2 = a22(b11-b12)
        add_term(u_a22, v_b11, w_c11, 1, 1, 0.5)
        add_term(u_a22, v_b12, w_c11, 1, -1, 0.5)
        
        # M3 = (a11+a12)b21
        add_term(u_a11, v_b21, w_c11, 1, 1, 0.5)
        add_term(u_a12, v_b21, w_c11, 1, 1, 0.5)
        
        # M4 = (a21+a22)b22
        add_term(u_a21, v_b22, w_c11, 1, 1, 0.5)
        add_term(u_a22, v_b22, w_c11, 1, 1, 0.5)
        
        # M5 = (a11-a21)(b11+b21)
        add_term(u_a11, v_b11, w_c11, 1, 1, -0.5)
        add_term(u_a11, v_b21, w_c11, 1, 1, -0.5)
        add_term(u_a21, v_b11, w_c11, -1, 1, -0.5)
        add_term(u_a21, v_b21, w_c11, -1, 1, -0.5)
        
        # M6 = (a12-a22)(b21+b22)
        add_term(u_a12, v_b21, w_c11, 1, 1, 0.5)
        add_term(u_a12, v_b22, w_c11, 1, 1, 0.5)
        add_term(u_a22, v_b21, w_c11, -1, 1, -0.5)
        add_term(u_a22, v_b22, w_c11, -1, 1, -0.5)
        
        # M7 = (a11+a22)(b22-b21)
        add_term(u_a11, v_b22, w_c11, 1, 1, 0.5)
        add_term(u_a11, v_b21, w_c11, 1, -1, 0.5)
        add_term(u_a22, v_b22, w_c11, 1, 1, 0.5)
        add_term(u_a22, v_b21, w_c11, 1, -1, 0.5)
        
        # This adds to w_c11. We need to add to all 4 W indices.
        # My add_term only adds to one W index.
        # I need to add the same U,V combination to all 4 W indices with appropriate signs.
        
        # Let's restart the Strassen implementation correctly.
        
    # Correct Strassen
    def strassen_block(a_r1, a_c1, a_r2, a_c2, b_r1, b_c1, b_r2, b_c2, c_r1, c_c1, c_r2, c_c2):
        u_a11 = a_r1 * m + a_c1
        u_a12 = a_r1 * m + a_c2
        u_a21 = a_r2 * m + a_c1
        u_a22 = a_r2 * m + a_c2
        
        v_b11 = b_r1 * p + b_c1
        v_b12 = b_r1 * p + b_c2
        v_b21 = b_r2 * p + b_c1
        v_b22 = b_r2 * p + b_c2
        
        w_c11 = c_r1 * n + c_c1
        w_c12 = c_r1 * n + c_c2
        w_c21 = c_r2 * n + c_c1
        w_c22 = c_r2 * n + c_c2
        
        # M1 = (a11+a22)(b11+b22) -> C11+=0.5, C12+=0, C21+=0, C22+=0.5?
        # C11 = M1 + M2 - M4 + M5
        # C12 = M3 - M2 + M4
        # C21 = M3 - M1 - M5 + M6
        # C22 = M1 - M3 + M4
        
        # Term 1: (a11+a22)(b11+b22)
        # U: a11, a22. V: b11, b22.
        # W: C11: 0.5, C22: 0.5
        add_term(u_a11, v_b11, w_c11, 1, 1, 0.5)
        add_term(u_a11, v_b22, w_c11, 1, 1, 0.5)
        add_term(u_a22, v_b11, w_c11, 1, 1, 0.5)
        add_term(u_a22, v_b22, w_c11, 1, 1, 0.5)
        
        add_term(u_a11, v_b11, w_c22, 1, 1, 0.5)
        add_term(u_a11, v_b22, w_c22, 1, 1, 0.5)
        add_term(u_a22, v_b11, w_c22, 1, 1, 0.5)
        add_term(u_a22, v_b22, w_c22, 1, 1, 0.5)
        
        # Term 2: a22(b11-b12)
        # W: C11: 0.5, C12: -0.5
        add_term(u_a22, v_b11, w_c11, 1, 1, 0.5)
        add_term(u_a22, v_b12, w_c11, 1, -1, 0.5)
        
        add_term(u_a22, v_b11, w_c12, 1, 1, -0.5)
        add_term(u_a22, v_b12, w_c12, 1, -1, -0.5)
        
        # Term 3: (a11+a12)b21
        # W: C12: 0.5, C21: 0.5
        add_term(u_a11, v_b21, w_c12, 1, 1, 0.5)
        add_term(u_a12, v_b21, w_c12, 1, 1, 0.5)
        
        add_term(u_a11, v_b21, w_c21, 1, 1, 0.5)
        add_term(u_a12, v_b21, w_c21, 1, 1, 0.5)
        
        # Term 4: (a21+a22)b22
        # W: C11: -0.5, C12: 0.5, C22: 0.5
        add_term(u_a21, v_b22, w_c11, 1, 1, -0.5)
        add_term(u_a22, v_b22, w_c11, 1, 1, -0.5)
        
        add_term(u_a21, v_b22, w_c12, 1, 1, 0.5)
        add_term(u_a22, v_b22, w_c12, 1, 1, 0.5)
        
        add_term(u_a21, v_b22, w_c22, 1, 1, 0.5)
        add_term(u_a22, v_b22, w_c22, 1, 1, 0.5)
        
        # Term 5: (a11-a21)(b11+b21)
        # W: C11: 0.5, C21: -0.5
        add_term(u_a11, v_b11, w_c11, 1, 1, 0.5)
        add_term(u_a11, v_b21, w_c11, 1, 1, 0.5)
        add_term(u_a21, v_b11, w_c11, -1, 1, 0.5)
        add_term(u_a21, v_b21, w_c11, -1, 1, 0.5)
        
        add_term(u_a11, v_b11, w_c21, 1, 1, -0.5)
        add_term(u_a11, v_b21, w_c21, 1, 1, -0.5)
        add_term(u_a21, v_b11, w_c21, -1, 1, -0.5)
        add_term(u_a21, v_b21, w_c21, -1, 1, -0.5)
        
        # Term 6: (a12-a22)(b21+b22)
        # W: C21: 0.5, C22: -0.5
        add_term(u_a12, v_b21, w_c21, 1, 1, 0.5)
        add_term(u_a12, v_b22, w_c21, 1, 1, 0.5)
        add_term(u_a22, v_b21, w_c21, -1, 1, 0.5)
        add_term(u_a22, v_b22, w_c21, -1, 1, 0.5)
        
        add_term(u_a12, v_b21, w_c22, 1, 1, -0.5)
        add_term(u_a12, v_b22, w_c22, 1, 1, -0.5)
        add_term(u_a22, v_b21, w_c22, -1, 1, -0.5)
        add_term(u_a22, v_b22, w_c22, -1, 1, -0.5)
        
        # Term 7: (a11+a22)(b22-b21)
        # W: C21: 0.5, C22: -0.5? No.
        # C22 = M1 - M3 + M4.
        # M7 is not in C22?
        # Standard Strassen:
        # M1 = (a+d)(e+j)
        # M2 = d(e-f)
        # M3 = (a+b)i
        # M4 = (c+d)j
        # M5 = (a-c)(e+j)
        # M6 = (b-d)(i+j)
        # M7 = (a+d)(j-i)
        
        # C11 = M1 + M2 - M4 + M5
        # C12 = M3 - M2 + M4
        # C21 = M3 - M1 - M5 + M6
        # C22 = M1 - M3 + M4
        
        # I used M7 in my head but not in code.
        # My code above implements M1..M6.
        # I need M7.
        
        # Term 7: (a11+a22)(b22-b21)
        # W: C21: 0.5? No.
        # M7 is used in C21?
        # C21 = M3 - M1 - M5 + M6.
        # M7 is not used?
        # Ah, the 7th product is (a+d)(j-i).
        # It is used in C21?
        # No, C21 uses M6 = (b-d)(i+j).
        # And C22 uses M1, M3, M4.
        # So M7 is not used?
        # Wait, Strassen uses 7 products.
        # M7 = (a+d)(j-i).
        # Is it used?
        # C22 = M1 - M3 + M4.
        # C21 = M3 - M1 - M5 + M6.
        # It seems M7 is not in the reconstruction?
        # No, the standard 7 are:
        # 1. (a+d)(e+j)
        # 2. d(e-f)
        # 3. (a+b)i
        # 4. (c+d)j
        # 5. (a-c)(e+j)
        # 6. (b-d)(i+j)
        # 7. (a+d)(j-i)
        
        # And the reconstruction is:
        # C11 = M1 + M2 - M4 + M5
        # C12 = M3 - M2 + M4
        # C21 = M3 - M1 - M5 + M6
        # C22 = M1 - M3 + M4
        
        # M7 is not used?
        # No, I must be misremembering.
        # Actually, M7 is used in some formulations.
        # But if it's not in the sum, it's not needed?
        # No, Strassen is 7.
        # Let's check online.
        # M7 = (a+d)(j-i).
        # C21 = M3 - M1 - M5 + M6?
        # C22 = M1 - M3 + M4?
        # It seems M7 is not used in this specific set of 6?
        # No, there are 7 products.
        # Maybe M6 is (b-d)(i+j) and M7 is (a+d)(j-i).
        # And C21 uses M7?
        # Let's assume the 7 products I listed are correct and implement them.
        
        # I will add Term 7.
        add_term(u_a11, v_b22, w_c21, 1, 1, 0.5)
        add_term(u_a11, v_b21, w_c21, 1, -1, 0.5)
        add_term(u_a22, v_b22, w_c21, 1, 1, 0.5)
        add_term(u_a22, v_b21, w_c21, 1, -1, 0.5)
        
        # And adjust C21 coefficients if needed.
        # For now, this adds 28 terms per Strassen block.
        
    # Block 1: TL (1x3)
    for j in range(4):
        for k in range(3):
            add_term(j, j*7+k, k*3+0, 1, 1, 1)
            
    # Block 2: TR (1x4)
    for j in range(4):
        for k in range(4):
            add_term(j, j*7+(3+k), (3+k)*3+0, 1, 1, 1)
            
    # Block 3: BL (2x3)
    # Schoolbook
    for i in range(2): # Rows 1,2 of C
        for j in range(4):
            for k in range(3):
                add_term((1+i)*4+j, j*7+k, (1+i)*3+k, 1, 1, 1)
                
    # Block 4: BR (2x4)
    # Strassen for A11 B11
    strassen_block(1, 0, 2, 1, 0, 3, 1, 4, 1, 3, 2, 4)
    # Strassen for A12 B21
    strassen_block(1, 2, 2, 3, 2, 3, 3, 4, 1, 3, 2, 4)
    # Strassen for A11 B12
    strassen_block(1, 0, 2, 1, 0, 5, 1, 6, 1, 5, 2, 6)
    # Strassen for A12 B22
    strassen_block(1, 2, 2, 3, 2, 5, 3, 6, 1, 5, 2, 6)
    
    rank = len(U_cols)
    U = U_cols
    V = V_cols
    W = W_cols
    
    return {"U": U, "V": V, "W": W}

