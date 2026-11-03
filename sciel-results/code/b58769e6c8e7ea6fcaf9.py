import time
import sys

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    
    # We construct the decomposition as a direct sum of:
    # 1. Two M_{4,4,4} blocks (rank 49 each)
    # 2. One M_{4,4,1} block (rank 16)
    # Total rank = 49 + 49 + 16 = 114
    
    U = []
    V = []
    W = []
    
    def add_factor(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    # --- Helper: Get Strassen factors for 2x2x2 ---
    # Returns list of 7 tuples (u, v, w) where u,v,w are length 4 lists
    # u: coeffs for A (00, 01, 10, 11)
    # v: coeffs for B (00, 01, 10, 11)
    # w: coeffs for C (00, 01, 10, 11)
    
    def get_strassen_2x2_factors():
        # m1 = (a+d)(e+h)
        # u1 = [1, 0, 0, 1], v1 = [1, 0, 0, 1]
        # m1 contributes to p (00) with +1, s (11) with +1
        # w1 = [1, 0, 0, 1]
        
        # m2 = (c+d)e
        # u2 = [0, 0, 1, 1], v2 = [1, 0, 0, 0]
        # m2 contributes to r (10) with +1, s (11) with -1
        # w2 = [0, 0, 1, -1]
        
        # m3 = a(f-h)
        # u3 = [1, 0, 0, 0], v3 = [0, 1, 0, -1]
        # m3 contributes to q (01) with +1, s (11) with +1
        # w3 = [0, 1, 0, 1]
        
        # m4 = d(g-e)
        # u4 = [0, 0, 0, 1], v4 = [0, 0, 1, -1]
        # m4 contributes to p (00) with +1, r (10) with +1
        # w4 = [1, 0, 1, 0]
        
        # m5 = (a+b)h
        # u5 = [1, 1, 0, 0], v5 = [0, 0, 0, 1]
        # m5 contributes to p (00) with -1, q (01) with +1
        # w5 = [-1, 1, 0, 0]
        
        # m6 = (c-a)(e+f)
        # u6 = [-1, 0, 1, 0], v6 = [1, 1, 0, 0]
        # m6 contributes to s (11) with +1
        # w6 = [0, 0, 0, 1]
        
        # m7 = (b-d)(g+h)
        # u7 = [0, 1, 0, -1], v7 = [0, 0, 1, 1]
        # m7 contributes to p (00) with +1
        # w7 = [1, 0, 0, 0]
        
        return [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
    strassen_factors = get_strassen_2x2_factors()
    
    # --- Helper: Add 4x4x4 block decomposition ---
    # This function adds the 49 factors for a 4x4x4 multiplication.
    # It assumes the A block is at A[0:4, 0:4], B at B[0:4, 0:4] (conceptually),
    # and C at C[0:4, 0:4].
    # However, we are building global U, V, W vectors.
    # We need to map local 2x2 indices to global 4x4 indices.
    
    def add_4x4x4_block(a_row_offset, b_row_offset, c_col_offset):
        # a_row_offset: offset in A rows (0)
        # b_row_offset: offset in B rows (0)
        # c_col_offset: offset in C cols (0 or 4)
        
        # We use block Strassen.
        # Blocks of A: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
        # Blocks of B: B00 (rows 0-1, cols 0-1), etc.
        # Blocks of C: C00 (rows 0-1, cols 0-1), etc.
        
        # Define the 7 Strassen block products M1..M7
        # Each Mk is a product Xk * Yk
        # Xk is a linear combo of A blocks
        # Yk is a linear combo of B blocks
        
        # We represent the coefficients of Xk and Yk as 4x4 matrices (block level)
        # But actually, we just need the coefficients for each 2x2 block.
        
        # Let's define the block coefficients for Xk (A side) and Yk (B side)
        # Xk_blocks[i][j] is the coefficient of A_ij in Xk?
        # No, Xk is a 2x2 matrix. Its entries are linear combinations of A blocks.
        # Wait, Xk is formed from A blocks.
        # e.g. M1 = (A00 + A11) * (B00 + B11)
        # Here X1 = A00 + A11. This is a 2x2 matrix?
        # No, A00 is a 2x2 matrix. A11 is a 2x2 matrix.
        # So X1 is a 2x2 matrix where each entry is a linear combo of entries of A00 and A11?
        # Actually, the Strassen formula operates on the *blocks* as scalars?
        # No, it operates on the blocks as 2x2 matrices.
        # So X1 is a 2x2 matrix of *values*? No.
        # The formula $M_1 = (A_{00} + A_{11})(B_{00} + B_{11})$ means:
        # Compute the sum of the two 2x2 matrices $A_{00} + A_{11}$, and $B_{00} + B_{11}$,
        # and then multiply them using the 2x2x2 Strassen algorithm.
        
        # So, for each block product $M_k = X_k Y_k$:
        # 1. Determine the linear combination of A blocks that form $X_k$.
        # 2. Determine the linear combination of B blocks that form $Y_k$.
        # 3. Use the 2x2x2 Strassen factors to decompose $X_k Y_k$.
        
        # Let's define the block coefficients for Xk and Yk.
        # Xk is a 2x2 matrix. Let's say $X_k = \sum_{i,j} \alpha_{ij} A_{ij}$.
        # But $A_{ij}$ are 2x2 matrices.
        # So $X_k$ is a 2x2 matrix where each entry is a linear combination of entries of the A blocks.
        
        # Actually, it's simpler:
        # $X_k$ is a 2x2 matrix.
        # $Y_k$ is a 2x2 matrix.
        # The product $M_k = X_k Y_k$ is a 2x2 matrix.
        
        # We need to express $X_k$ in terms of the global A entries.
        # And $Y_k$ in terms of the global B entries.
        
        # Let's define the block coefficient matrices for Xk and Yk.
        # BlockCoeffX[i][j] is the coefficient of the block $A_{ij}$ in $X_k$?
        # No, $X_k$ is a single 2x2 matrix.
        # $X_k = \alpha_{00} A_{00} + \alpha_{01} A_{01} + \alpha_{10} A_{10} + \alpha_{11} A_{11}$.
        # Here $\alpha_{ij}$ are scalars (0 or 1).
        
        # For M1: X1 = A00 + A11. So alpha_00=1, alpha_11=1, others 0.
        # Y1 = B00 + B11. So beta_00=1, beta_11=1, others 0.
        
        # For M2: X2 = A10 + A11. alpha_10=1, alpha_11=1.
        # Y2 = B00. beta_00=1.
        
        # And so on.
        
        block_defs = [
            # (X_coeffs, Y_coeffs)
            # X_coeffs: dict {(i,j): coeff}
            # Y_coeffs: dict {(i,j): coeff}
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),           # M2
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),          # M3
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),          # M4
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),           # M5
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),# M6
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}) # M7
        ]
        
        # C block contributions for M1..M7
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        c_contribs = [
            {(0,0): 1, (1,1): 1},  # M1
            {(1,0): 1, (1,1): -1}, # M2
            {(0,1): 1, (1,1): 1},  # M3
            {(0,0): 1, (1,0): 1},  # M4
            {(0,0): -1, (0,1): 1}, # M5
            {(1,1): 1},            # M6
            {(0,0): 1}             # M7
        ]
        
        for k in range(7):
            x_coeffs = block_defs[k][0]
            y_coeffs = block_defs[k][1]
            c_coeffs = c_contribs[k]
            
            # For each of the 7 Strassen factors of the 2x2x2 product:
            for (u2, v2, w2) in strassen_factors:
                # u2 is length 4: coeffs for X2 entries (00, 01, 10, 11)
                # v2 is length 4: coeffs for Y2 entries (00, 01, 10, 11)
                # w2 is length 4: coeffs for Mk entries (00, 01, 10, 11)
                
                # Construct global U vector (length 16)
                # Global A index: i*4 + j.
                # Block A_ij contains entries at rows i*2..i*2+1, cols j*2..j*2+1.
                # The 2x2 Strassen factor u2[r] applies to the r-th entry of the 2x2 matrix Xk.
                # The r-th entry of Xk is a linear combo of entries of A blocks.
                
                # Let's map local 2x2 indices to global 4x4 indices.
                # Local 2x2 index (lr, lc) corresponds to:
                # If Xk = sum alpha_ab A_ab, then Xk[lr, lc] = sum alpha_ab A_ab[lr, lc].
                # A_ab[lr, lc] is at global index (a*2+lr)*4 + (b*2+lc).
                
                u_global = [0] * 16
                for lr in range(2):
                    for lc in range(2):
                        coeff = u2[lr*2 + lc]
                        if coeff == 0: continue
                        for (a, b), alpha in x_coeffs.items():
                            if alpha == 0: continue
                            global_i = a*2 + lr
                            global_j = b*2 + lc
                            u_global[global_i*4 + global_j] += coeff * alpha
                
                v_global = [0] * 36 # m*p = 4*9
                # B blocks are at rows 0..3, cols 0..3 (for this block)
                # Global B index: j*9 + k.
                # Block B_cd contains entries at rows c*2..c*2+1, cols d*2..d*2+1.
                # Yk[lr, lc] = sum beta_cd B_cd[lr, lc].
                # B_cd[lr, lc] is at global index (c*2+lr)*9 + (d*2+lc).
                # Note: The B block for this 4x4x4 multiplication uses cols 0..3 (if c_col_offset=0) or 4..7 (if c_col_offset=4).
                # Wait, the B matrix is 4x9.
                # The first 4x4x4 block uses B cols 0..3.
                # The second 4x4x4 block uses B cols 4..7.
                # So we need to add c_col_offset to the B column index.
                
                for lr in range(2):
                    for lc in range(2):
                        coeff = v2[lr*2 + lc]
                        if coeff == 0: continue
                        for (c, d), beta in y_coeffs.items():
                            if beta == 0: continue
                            global_j = c*2 + lr
                            global_k = d*2 + lc + c_col_offset
                            v_global[global_j*9 + global_k] += coeff * beta
                
                w_global = [0] * 36 # p*n = 9*4
                # C blocks are at rows 0..3, cols 0..3 (for this block)
                # Global C index: k*4 + i.
                # Block C_ef contains entries at rows e*2..e*2+1, cols f*2..f*2+1.
                # Mk[lr, lc] contributes to C_ef[lr, lc] with coeff w2[lr, lc] * gamma_ef.
                # C_ef[lr, lc] is at global index (f*2+lc)*4 + (e*2+lr).
                # Note: The C block for this multiplication uses cols 0..3 (if c_col_offset=0) or 4..7 (if c_col_offset=4).
                
                for lr in range(2):
                    for lc in range(2):
                        coeff = w2[lr*2 + lc]
                        if coeff == 0: continue
                        for (e, f), gamma in c_coeffs.items():
                            if gamma == 0: continue
                            global_i = e*2 + lr
                            global_k = f*2 + lc + c_col_offset
                            w_global[global_k*4 + global_i] += coeff * gamma
                
                add_factor(u_global, v_global, w_global)
                
    # Add first 4x4x4 block (cols 0-3 of B)
    add_4x4x4_block(0, 0, 0)
    
    # Add second 4x4x4 block (cols 4-7 of B)
    add_4x4x4_block(0, 0, 4)
    
    # Add 4x4x1 block (col 8 of B)
    # Rank 16
    # C_i = sum_j A_ij B_j8
    for i in range(4):
        for j in range(4):
            u = [0] * 16
            v = [0] * 36
            w = [0] * 36
            
            u[i*4 + j] = 1
            v[j*9 + 8] = 1
            w[8*4 + i] = 1
            
            add_factor(u, v, w)
            
    # Convert lists of lists to lists of lists (already done)
    # U is list of 16-length lists
    # V is list of 36-length lists
    # W is list of 36-length lists
    
    return {"U": U, "V": V, "W": W}

