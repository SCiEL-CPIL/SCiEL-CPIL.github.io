"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 2 x 2 x 8 blocks (using Strassen 4 times) -> 4 * 7 = 28 each. Two such blocks -> 56.
    # 1 x 2 x 8 blocks (schoolbook) -> 16 each. Two such blocks -> 32.
    # Total rank = 88.
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        if col >= rank:
            raise Exception("Rank overflow")
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Strassen factors for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, k_offset):
        """
        Adds Strassen decomposition for a 2x2 A block, 2x2 B block (for 2 output cols), 
        contributing to 2x2 C block.
        a_rows: [r1, r2]
        a_cols: [c1, c2]
        b_rows: [r1, r2]
        b_cols: [c1, c2] (these are the k indices for the 2x2 B block)
        c_rows: [r1, r2]
        c_cols: [c1, c2]
        """
        # Local indices 0,1 for a,b,c,d and e,f,g,h
        # a = A[r1][c1], b = A[r1][c2], c = A[r2][c1], d = A[r2][c2]
        # e = B[r1][k1], f = B[r1][k2], g = B[r2][k1], h = B[r2][k2]
        # C11 = C[r1][k1], C12 = C[r1][k2], C21 = C[r2][k1], C22 = C[r2][k2]
        
        u_a = a_rows[0] * m + a_cols[0]
        u_b = a_rows[0] * m + a_cols[1]
        u_c = a_rows[1] * m + a_cols[0]
        u_d = a_rows[1] * m + a_cols[1]
        
        v_e = b_rows[0] * p + b_cols[0]
        v_f = b_rows[0] * p + b_cols[1]
        v_g = b_rows[1] * p + b_cols[0]
        v_h = b_rows[1] * p + b_cols[1]
        
        w_c11 = b_cols[0] * n + c_rows[0]
        w_c12 = b_cols[1] * n + c_rows[0]
        w_c21 = b_cols[0] * n + c_rows[1]
        w_c22 = b_cols[1] * n + c_rows[1]
        
        # M1 = (a+d)(e+h)
        add_term(u_a, 1, v_e, 1, w_c11, 1)
        add_term(u_d, 1, v_e, 1, w_c11, 0) # Wait, M1 is one term. 
        # Correction: add_term adds a full rank-1 component.
        # M1 is (a+d)(e+h). This is not a single rank-1 term in U,V,W unless we expand.
        # (a+d)(e+h) = ae + ah + de + dh.
        # This requires 4 rank-1 terms if we don't combine.
        # But Strassen combines them into 1 multiplication.
        # In tensor decomposition, a "multiplication" $x y$ corresponds to a rank-1 term $U_x V_y W_{xy}$.
        # If $x = a+d$, $y = e+h$, then the term is $(a+d)(e+h) W$.
        # This can be represented as:
        # U_a += 1, U_d += 1
        # V_e += 1, V_h += 1
        # W += 1
        # This is a SINGLE rank-1 term in the decomposition sense?
        # No. The rank is the number of columns.
        # If we use 1 column, we set U_a=1, U_d=1, V_e=1, V_h=1, W=1.
        # The product is $(1\cdot A_a + 1\cdot A_d)(1\cdot B_e + 1\cdot B_h) W = (A_a+A_d)(B_e+B_h)W$.
        # This is correct.
        # So each Strassen factor M_i corresponds to ONE column in U, V, W.
        
        # Let's restart the add_term logic for Strassen.
        # Each M_i uses 1 rank unit.
        
        # M1
        add_term(u_a, 1, v_e, 1, w_c11, 1) # This is wrong. add_term increments col.
        # I need a function that writes to the CURRENT column and then increments.
        
        # Let's redefine add_term to just write to the current col index passed or use nonlocal col.
        # The previous add_term was:
        # def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        #     nonlocal col
        #     U[u_idx][col] += u_val
        #     V[v_idx][col] += v_val
        #     W[w_idx][col] += w_val
        #     col += 1
        # This adds ONE rank-1 term.
        # For M1 = (a+d)(e+h), we need to add coefficients to U, V, W for the SAME column.
        # So we should NOT increment col for each part of the factor.
        # We should increment col ONCE per Strassen factor.
        
        # Revised plan:
        # Define a function add_strassen_factor(u_coeffs, v_coeffs, w_coeffs)
        # where u_coeffs is dict {idx: val}, etc.
        
        pass

    # Let's implement the correct addition logic.
    
    def add_rank_unit(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            raise Exception("Rank overflow")
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1

    # Strassen Block 1: A[0:2, 0:2] * B[0:2, 0:8] -> C[0:2, 0:8]
    # Split B into 4 blocks of 2x2.
    for k_start in range(0, 8, 2):
        k1 = k_start
        k2 = k_start + 1
        a_rows = [0, 1]
        a_cols = [0, 1]
        b_rows = [0, 1]
        b_cols = [k1, k2]
        c_rows = [0, 1]
        c_cols = [k1, k2]
        
        u_a = a_rows[0] * m + a_cols[0]
        u_b = a_rows[0] * m + a_cols[1]
        u_c = a_rows[1] * m + a_cols[0]
        u_d = a_rows[1] * m + a_cols[1]
        
        v_e = b_rows[0] * p + b_cols[0]
        v_f = b_rows[0] * p + b_cols[1]
        v_g = b_rows[1] * p + b_cols[0]
        v_h = b_rows[1] * p + b_cols[1]
        
        w_c11 = b_cols[0] * n + c_rows[0]
        w_c12 = b_cols[1] * n + c_rows[0]
        w_c21 = b_cols[0] * n + c_rows[1]
        w_c22 = b_cols[1] * n + c_rows[1]
        
        # M1 = (a+d)(e+h)
        add_rank_unit({u_a: 1, u_d: 1}, {v_e: 1, v_h: 1}, {w_c11: 1, w_c22: 1})
        # M2 = (c+d)e
        add_rank_unit({u_c: 1, u_d: 1}, {v_e: 1}, {w_c21: 1, w_c22: -1})
        # M3 = a(f-h)
        add_rank_unit({u_a: 1}, {v_f: 1, v_h: -1}, {w_c12: 1, w_c22: 1})
        # M4 = d(g-e)
        add_rank_unit({u_d: 1}, {v_g: 1, v_e: -1}, {w_c11: 1, w_c21: 1})
        # M5 = (a+b)h
        add_rank_unit({u_a: 1, u_b: 1}, {v_h: 1}, {w_c11: -1, w_c12: 1})
        # M6 = (c-a)(e+f)
        add_rank_unit({u_c: 1, u_a: -1}, {v_e: 1, v_f: 1}, {w_c22: 1})
        # M7 = (b-d)(g+h)
        add_rank_unit({u_b: 1, u_d: -1}, {v_g: 1, v_h: 1}, {w_c11: 1})
        
    # Strassen Block 3: A[0:2, 2:4] * B[2:4, 0:8] -> C[0:2, 0:8]
    for k_start in range(0, 8, 2):
        k1 = k_start
        k2 = k_start + 1
        a_rows = [0, 1]
        a_cols = [2, 3]
        b_rows = [2, 3]
        b_cols = [k1, k2]
        c_rows = [0, 1]
        c_cols = [k1, k2]
        
        u_a = a_rows[0] * m + a_cols[0]
        u_b = a_rows[0] * m + a_cols[1]
        u_c = a_rows[1] * m + a_cols[0]
        u_d = a_rows[1] * m + a_cols[1]
        
        v_e = b_rows[0] * p + b_cols[0]
        v_f = b_rows[0] * p + b_cols[1]
        v_g = b_rows[1] * p + b_cols[0]
        v_h = b_rows[1] * p + b_cols[1]
        
        w_c11 = b_cols[0] * n + c_rows[0]
        w_c12 = b_cols[1] * n + c_rows[0]
        w_c21 = b_cols[0] * n + c_rows[1]
        w_c22 = b_cols[1] * n + c_rows[1]
        
        # M1 = (a+d)(e+h)
        add_rank_unit({u_a: 1, u_d: 1}, {v_e: 1, v_h: 1}, {w_c11: 1, w_c22: 1})
        # M2 = (c+d)e
        add_rank_unit({u_c: 1, u_d: 1}, {v_e: 1}, {w_c21: 1, w_c22: -1})
        # M3 = a(f-h)
        add_rank_unit({u_a: 1}, {v_f: 1, v_h: -1}, {w_c12: 1, w_c22: 1})
        # M4 = d(g-e)
        add_rank_unit({u_d: 1}, {v_g: 1, v_e: -1}, {w_c11: 1, w_c21: 1})
        # M5 = (a+b)h
        add_rank_unit({u_a: 1, u_b: 1}, {v_h: 1}, {w_c11: -1, w_c12: 1})
        # M6 = (c-a)(e+f)
        add_rank_unit({u_c: 1, u_a: -1}, {v_e: 1, v_f: 1}, {w_c22: 1})
        # M7 = (b-d)(g+h)
        add_rank_unit({u_b: 1, u_d: -1}, {v_g: 1, v_h: 1}, {w_c11: 1})
        
    # Schoolbook Block 2: A[2:3, 0:2] * B[0:2, 0:8] -> C[2:3, 0:8]
    # C[2,k] = A[2,0]B[0,k] + A[2,1]B[1,k]
    for k in range(8):
        # Term 1: A[2,0] B[0,k]
        u_idx = 2 * m + 0
        v_idx = 0 * p + k
        w_idx = k * n + 2
        add_rank_unit({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
        
        # Term 2: A[2,1] B[1,k]
        u_idx = 2 * m + 1
        v_idx = 1 * p + k
        w_idx = k * n + 2
        add_rank_unit({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
        
    # Schoolbook Block 4: A[2:3, 2:4] * B[2:4, 0:8] -> C[2:3, 0:8]
    # C[2,k] = A[2,2]B[2,k] + A[2,3]B[3,k]
    for k in range(8):
        # Term 1: A[2,2] B[2,k]
        u_idx = 2 * m + 2
        v_idx = 2 * p + k
        w_idx = k * n + 2
        add_rank_unit({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
        
        # Term 2: A[2,3] B[3,k]
        u_idx = 2 * m + 3
        v_idx = 3 * p + k
        w_idx = k * n + 2
        add_rank_unit({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
