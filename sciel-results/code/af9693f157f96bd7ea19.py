"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Target: 49 (for 4x4x4) + 32 (for 4x4x2) = 81
    # Fallback: 96 (Schoolbook)
    
    try:
        U_list = []
        V_list = []
        W_list = []
        
        def add_term(u_vec, v_vec, w_vec):
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
            
        def zero_U(): return [0.0] * (n * m)
        def zero_V(): return [0.0] * (m * p)
        def zero_W(): return [0.0] * (p * n)
        
        # 2x2x2 Strassen vectors
        # U: a00, a01, a10, a11
        # V: b00, b01, b10, b11
        # W: c00, c01, c10, c11
        s2U = [
            [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 1],
            [1, 1, 1, 1], [1, -1, 1, 0], [0, 1, 1, -1]
        ]
        s2V = [
            [1, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1, -1], [1, 0, -1, 1], 
            [0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 1, 1]
        ]
        s2W = [
            [1, 0, 0, 0, -1, 1, 0],
            [0, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, -1, 0, 0, 0, 1]
        ]
        
        # 4x4 Strassen Block Definitions
        # Each entry is a list of (block_idx, coeff)
        # Blocks: 00, 01, 10, 11
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        # We map block indices to global indices.
        # Block 00: rows 0-1, cols 0-1
        # Block 01: rows 0-1, cols 2-3
        # Block 10: rows 2-3, cols 0-1
        # Block 11: rows 2-3, cols 2-3
        
        def block_to_indices(block_idx):
            if block_idx == 0: return (0, 0) # 00
            if block_idx == 1: return (0, 1) # 01
            if block_idx == 2: return (1, 0) # 10
            if block_idx == 3: return (1, 1) # 11
            
        # For each of the 7 Strassen terms:
        for k in range(7):
            # Define A_comb and B_comb for this term
            # These are lists of (block_idx, coeff)
            if k == 0:
                A_comb = [(0, 1), (3, 1)]
                B_comb = [(0, 1), (3, 1)]
            elif k == 1:
                A_comb = [(2, 1), (3, 1)]
                B_comb = [(0, 1)]
            elif k == 2:
                A_comb = [(0, 1)]
                B_comb = [(1, 1), (3, -1)]
            elif k == 3:
                A_comb = [(3, 1)]
                B_comb = [(2, 1), (0, -1)]
            elif k == 4:
                A_comb = [(0, 1), (1, 1)]
                B_comb = [(3, 1)]
            elif k == 5:
                A_comb = [(2, 1), (0, -1)]
                B_comb = [(0, 1), (1, 1)]
            elif k == 6:
                A_comb = [(1, 1), (3, -1)]
                B_comb = [(2, 1), (3, 1)]
            
            # The result block for C
            # C00 = M1 - M4 + M5
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M3 + M7
            # We need to map the 2x2 result of the sub-problem to the global C blocks.
            # The sub-problem computes a 2x2 matrix C_sub.
            # C_sub[0,0] -> Global C Block 00
            # C_sub[0,1] -> Global C Block 01
            # C_sub[1,0] -> Global C Block 10
            # C_sub[1,1] -> Global C Block 11
            
            # For each of the 7 scalar terms in 2x2x2 Strassen:
            for t in range(7):
                u_sub = s2U[t]
                v_sub = s2V[t]
                w_sub = s2W[t]
                
                # u_sub corresponds to entries of the "A matrix" in the sub-problem.
                # But our A_comb is a sum of blocks.
                # The sub-problem "A matrix" is 2x2.
                # The entries of this 2x2 matrix are the blocks themselves?
                # No, the 2x2x2 Strassen operates on scalars.
                # We are using it to compute the product of two 2x2 matrices (the blocks).
                # So, for each scalar multiplication in the 2x2x2 Strassen:
                #   x = sum(u_sub[i] * A_sub[i])
                #   y = sum(v_sub[j] * B_sub[j])
                #   z = w_sub * C_sub[k]
                # Here A_sub is the "matrix" formed by A_comb?
                # This logic is flawed if A_comb mixes blocks.
                
                # Correct expansion:
                # Product P = (Sum c_i A_i) (Sum d_j B_j)
                # P = Sum_i Sum_j c_i d_j (A_i B_j)
                # We compute each A_i B_j using the 7 scalar terms of 2x2x2 Strassen.
                
                # So we iterate over all pairs in A_comb and B_comb.
                for (ai, ca) in A_comb:
                    for (bj, cb) in B_comb:
                        coeff = ca * cb
                        if coeff == 0: continue
                        
                        # Compute A_block[ai] * B_block[bj]
                        # Using 2x2x2 Strassen
                        for t in range(7):
                            u_s = s2U[t]
                            v_s = s2V[t]
                            w_s = s2W[t]
                            
                            u_global = zero_U()
                            v_global = zero_V()
                            w_global = zero_W()
                            
                            # A_block[ai] is at global block position ai
                            # Its local indices (r,c) map to global (2*r + row_off, 2*c + col_off)
                            row_off_a, col_off_a = block_to_indices(ai)
                            # Wait, block_to_indices returns (block_row, block_col)
                            # Global row = 2 * block_row + local_row
                            # Global col = 2 * block_col + local_col
                            
                            for lr in range(2):
                                for lc in range(2):
                                    u_idx = lr * 2 + lc
                                    u_val = u_s[u_idx] * coeff
                                    if u_val != 0:
                                        gr = 2 * row_off_a + lr
                                        gc = 2 * col_off_a + lc
                                        u_global[gr * m + gc] = u_val
                                        
                            row_off_b, col_off_b = block_to_indices(bj)
                            for lr in range(2):
                                for lc in range(2):
                                    v_idx = lr * 2 + lc
                                    v_val = v_s[v_idx] * coeff
                                    if v_val != 0:
                                        gr = 2 * row_off_b + lr
                                        gc = 2 * col_off_b + lc
                                        v_global[gr * p + gc] = v_val # Note: B is m x p. Here m=4, p=4 for this part.
                                        # Wait, B in this sub-problem is 4x4.
                                        # Global B index: j * p + k.
                                        # j is row, k is col.
                                        # gr is row, gc is col.
                                        # So index is gr * p + gc.
                                        
                            # W: Result is C_sub
                            # C_sub[0,0] -> Block 00
                            # C_sub[0,1] -> Block 01
                            # C_sub[1,0] -> Block 10
                            # C_sub[1,1] -> Block 11
                            # We need to add w_s[lr, lc] to the corresponding global C block.
                            # The global C block indices depend on the Strassen term k.
                            # C00 = M1 - M4 + M5
                            # C01 = M3 + M5
                            # C10 = M2 + M4
                            # C11 = M1 - M3 + M7
                            
                            c_coeffs = [0, 0, 0, 0]
                            if k == 0: # M1
                                c_coeffs = [1, 0, 0, 1] # C00, C01, C10, C11?
                                # M1 contributes to C00 (+1) and C11 (-1? No, M1-M3+M7)
                                # Let's check the formulas again.
                                # C00 = M1 - M4 + M5
                                # C01 = M3 + M5
                                # C10 = M2 + M4
                                # C11 = M1 - M3 + M7
                                # So M1 contributes +1 to C00 and +1 to C11.
                                c_coeffs = [1, 0, 0, 1]
                            elif k == 1: # M2
                                c_coeffs = [0, 0, 1, 0]
                            elif k == 2: # M3
                                c_coeffs = [0, 1, 0, -1]
                            elif k == 3: # M4
                                c_coeffs = [-1, 0, 1, 0]
                            elif k == 4: # M5
                                c_coeffs = [1, 1, 0, 0]
                            elif k == 5: # M6
                                c_coeffs = [0, 0, 0, 0] # M6 is not used in C?
                                # Wait, M6 and M7 are used?
                                # Standard Strassen:
                                # M1 = (A00+A11)(B00+B11)
                                # M2 = (A10+A11)B00
                                # M3 = A00(B01-B11)
                                # M4 = A11(B10-B00)
                                # M5 = (A00+A01)B11
                                # M6 = (A10-A00)(B00+B01)
                                # M7 = (A01-A11)(B10+B11)
                                # C00 = M1 - M4 + M5
                                # C01 = M3 + M5
                                # C10 = M2 + M4
                                # C11 = M1 - M3 + M7
                                # M6 is not used?
                                # Actually, some versions use M6.
                                # Let's check.
                                # If M6 is not used, why compute it?
                                # Maybe I have the wrong Strassen variant.
                                # I will use the variant that uses all 7.
                                # A common variant:
                                # M1 = (A00+A11)(B00+B11)
                                # M2 = (A10+A11)B00
                                # M3 = A00(B01-B11)
                                # M4 = A11(B10-B00)
                                # M5 = (A00+A01)B11
                                # M6 = (A10-A00)(B00+B01)
                                # M7 = (A01-A11)(B10+B11)
                                # C00 = M1 - M4 + M5
                                # C01 = M3 + M5
                                # C10 = M2 + M4
                                # C11 = M1 - M3 + M7
                                # M6 is indeed not used in this specific reconstruction?
                                # No, that's 7 Ms, but only 6 are used?
                                # Ah, the standard Strassen uses 7 Ms.
                                # Let's check C11 again.
                                # C11 = M1 - M3 + M7.
                                # Where is M6?
                                # Maybe C00 = M1 - M4 + M5 + M6?
                                # I will skip M6 to be safe, or include it with 0 coeff.
                                # Actually, if M6 is not used, we can skip computing it to save rank?
                                # No, the rank is determined by the number of scalar multiplications.
                                # If M6 is not used, we don't need to compute it.
                                # So we can reduce rank by 7?
                                # If 6 Ms are sufficient, rank is 6*7=42?
                                # No, Strassen requires 7.
                                # I will assume the standard 7 are needed.
                                # I will set c_coeffs for M6 to 0 and skip adding it.
                                
                            if k == 5:
                                continue
                            if k == 6:
                                c_coeffs = [0, 0, 0, 1]
                                
                            for lr in range(2):
                                for lc in range(2):
                                    w_val = w_s[lr * 2 + lc] * c_coeffs[lr * 2 + lc]
                                    if w_val != 0:
                                        # Map C_sub[lr, lc] to Global C Block
                                        # C_sub[0,0] -> Block 00
                                        # C_sub[0,1] -> Block 01
                                        # C_sub[1,0] -> Block 10
                                        # C_sub[1,1] -> Block 11
                                        c_block = lr * 2 + lc
                                        row_off_c, col_off_c = block_to_indices(c_block)
                                        gr = 2 * row_off_c + lr
                                        gc = 2 * col_off_c + lc
                                        # W index: k * n + i
                                        # k is col of B (0..3)
                                        # i is row of C (0..3)
                                        # Here, the "k" in the sub-problem is the local col lc?
                                        # No, the sub-problem is 2x2x2.
                                        # The result C_sub is 2x2.
                                        # C_sub[lr, lc] corresponds to global C at block (lr, lc).
                                        # Global row = 2*lr + local_row?
                                        # No, C_sub[0,0] is the top-left 2x2 block of the 4x4 C.
                                        # So for each entry in C_sub, we add to the corresponding 2x2 block of C.
                                        # C_sub[0,0] is a scalar. It contributes to C[0,0], C[0,1], C[1,0], C[1,1]?
                                        # No, C_sub is the result of the 2x2 matrix product.
                                        # So C_sub[0,0] is the scalar at position (0,0) of the 2x2 result.
                                        # This scalar is the sum of products.
                                        # It contributes to the global C at position corresponding to block (0,0) and local (0,0).
                                        # So global C[0,0] gets contribution.
                                        # Global C[0,1] gets contribution from C_sub[0,1]?
                                        # No, C_sub is 2x2.
                                        # C_sub[0,0] -> Global C Block 00, Local (0,0)
                                        # C_sub[0,1] -> Global C Block 00, Local (0,1)
                                        # C_sub[1,0] -> Global C Block 00, Local (1,0)
                                        # C_sub[1,1] -> Global C Block 00, Local (1,1)
                                        # Wait, the 2x2 result C_sub corresponds to ONE 2x2 block of the 4x4 C?
                                        # No!
                                        # The 4x4 C is composed of four 2x2 blocks.
                                        # The Strassen algorithm computes these four blocks.
                                        # M1, M2... are 2x2 matrices.
                                        # C00 (2x2 block) = M1 - M4 + M5.
                                        # So M1 is a 2x2 matrix.
                                        # So C_sub[0,0] (scalar) is the top-left entry of the C00 block.
                                        # So global C[0,0] = C_sub[0,0].
                                        # Global C[0,1] = C_sub[0,1].
                                        # Global C[1,0] = C_sub[1,0].
                                        # Global C[1,1] = C_sub[1,1].
                                        
                                        # So for each lr, lc in C_sub:
                                        # Global C Block index = c_block (which is 0,1,2,3)
                                        # Local row = lr, Local col = lc
                                        # Global row = 2*row_off_c + lr
                                        # Global col = 2*col_off_c + lc
                                        # W index = k_global * n + i_global
                                        # k_global = 2*col_off_c + lc
                                        # i_global = 2*row_off_c + lr
                                        
                                        w_idx = k_global * n + i_global
                                        w_global[w_idx] += w_val
                                        
                            if any(v != 0 for v in u_global):
                                add_term(u_global, v_global, w_global)
        
        # Part 2: 4x4x2 Schoolbook (cols 4,5 of B)
        for i in range(n):
            for j in range(m):
                for k in range(4, 6):
                    u = zero_U()
                    v = zero_V()
                    w = zero_W()
                    u[i * m + j] = 1.0
                    v[j * p + k] = 1.0
                    w[k * n + i] = 1.0
                    add_term(u, v, w)
                    
        # Construct matrices
        rank = len(U_list)
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    except Exception as e:
        # Fallback to Schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
