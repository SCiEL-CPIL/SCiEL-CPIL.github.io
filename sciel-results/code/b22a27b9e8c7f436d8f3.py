import numpy as np

def construct():
    n, m, p = 3, 4, 4
    
    # Fallback to current solution if Laderman fails
    def get_current_solution():
        rank = 44
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        
        # Strassen part (cols 0-6)
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1
        
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1
        
        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1
        
        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1
        
        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1
        
        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1
        
        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1
        
        # Extra part (cols 7-10)
        U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
        U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
        U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
        U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
        
        def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
            col_offset = block_idx * 11
            for r in range(3):
                for c in range(2):
                    local_a_idx = r * 2 + c
                    global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                    U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
            for r in range(2):
                for c in range(2):
                    local_b_idx = r * 2 + c
                    global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                    V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
            for r in range(3):
                for c in range(2):
                    local_c_idx = r * 2 + c
                    global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                    W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

        add_block(0, 0, 0, 0, 0, 0, 0)
        add_block(1, 0, 2, 2, 0, 0, 0)
        add_block(2, 0, 0, 0, 2, 0, 2)
        add_block(3, 0, 2, 2, 2, 0, 2)
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try Laderman
    try:
        rank_4x4 = 19
        U4 = np.zeros((16, rank_4x4))
        V4 = np.zeros((16, rank_4x4))
        W4 = np.zeros((16, rank_4x4))
        
        # Helper to add linear combination
        def add_factor(idx, a_coeffs, b_coeffs, c_coeffs):
            # a_coeffs: dict {(row, col): val}
            # b_coeffs: dict {(row, col): val}
            # c_coeffs: dict {(row, col): val}
            for (r, c), val in a_coeffs.items():
                U4[r * 4 + c, idx] += val
            for (r, c), val in b_coeffs.items():
                V4[r * 4 + c, idx] += val
            for (r, c), val in c_coeffs.items():
                W4[r * 4 + c, idx] += val

        # Laderman's 19 factors
        # A blocks: A11(0,0), A12(0,2), A21(2,0), A22(2,2)
        # B blocks: B11(0,0), B12(0,2), B21(2,0), B22(2,2)
        
        # 1. A11 B11
        add_factor(0, 
            {(0,0):1, (0,1):1, (1,0):1, (1,1):1}, 
            {(0,0):1, (0,1):1, (1,0):1, (1,1):1}, 
            {(0,0):1, (0,1):1, (1,0):1, (1,1):1})
            
        # 2. A22 B22
        add_factor(1, 
            {(2,0):1, (2,1):1, (3,0):1, (3,1):1}, 
            {(2,0):1, (2,1):1, (3,0):1, (3,1):1}, 
            {(2,0):1, (2,1):1, (3,0):1, (3,1):1})
            
        # 3. (A11+A22)(B11+B22)
        add_factor(2, 
            {(0,0):1, (0,1):1, (1,0):1, (1,1):1, (2,0):1, (2,1):1, (3,0):1, (3,1):1}, 
            {(0,0):1, (0,1):1, (1,0):1, (1,1):1, (2,0):1, (2,1):1, (3,0):1, (3,1):1}, 
            {}) # C is computed later? No, W is the output factor.
            # For rank decomposition, W[i, k] is the coefficient of the product in C[i,k].
            # C = sum P_k * W[:, k]
            # So W4 should contain the C block corresponding to the product.
            # For P3, C11 gets (A11+A22)(B11+B22) - A11B11 - A22B22?
            # No, the decomposition is C = sum U[:,k] * V[:,k] * W[:,k] ?
            # No, the tensor is T = sum_k U_k \otimes V_k \otimes W_k.
            # C_{ik} = sum_j A_{ij} B_{jk}.
            # T_{ijk} = delta_{...}
            # The factors U, V, W are such that T = sum_k U[:,k] V[:,k]^T W[:,k]?
            # No, U is (nm, rank), V is (mp, rank), W is (pn, rank).
            # T = sum_k U[:,k] \otimes V[:,k] \otimes W[:,k].
            # This means T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
            # Here i is index of A, j is index of B, k is index of C.
            # A index: row_A * 4 + col_A.
            # B index: row_B * 4 + col_B.
            # C index: col_C * 4 + row_C.
            
            # So for each factor r, we define linear combinations of A entries (U), B entries (V), and C entries (W).
            # The product of the linear combination of A and B must equal the linear combination of C?
            # No, the tensor equation is:
            # sum_{r} U[i,r] V[j,r] W[k,r] = T[i,j,k]
            # This is not "Product of sums = Sum of products".
            # It is a bilinear decomposition.
            # Actually, for matrix multiplication, the standard decomposition is:
            # C = sum_{r} (U_r A) * (V_r B)^T ? No.
            # The factors are vectors.
            # U[i,r] is coeff of A_i in the r-th linear form.
            # V[j,r] is coeff of B_j in the r-th linear form.
            # W[k,r] is coeff of C_k in the r-th linear form.
            # The condition is that the tensor T can be written as sum_r u_r \otimes v_r \otimes w_r.
            # This implies that for any A, B, C, if C = AB, then sum_r (u_r^T A) (v_r^T B) (w_r^T C) = 0? No.
            # The condition is that the matrix multiplication map is a sum of rank-1 tensors.
            # T = sum_r u_r \otimes v_r \otimes w_r.
            # This means T_{ijk} = sum_r U_{i,r} V_{j,r} W_{k,r}.
            
            # So I need to find U, V, W such that this holds.
            # For Laderman, the 19 products are intermediate scalars.
            # Let M_r be the r-th scalar product.
            # M_r = (sum_i U_{i,r} A_i) * (sum_j V_{j,r} B_j).
        # Given the risk of incorrect hardcoding, I will stick to the current solution or try a simpler optimization.
        # However, the directive strongly hints at the 4x4x4 rank-19.
        # I will try to implement a simple verification of a known rank-19 decomposition if I can find one online or derive it.
        # Since I cannot browse, I will rely on the current solution which is valid.
        
        return get_current_solution()
        
    except Exception as e:
        return get_current_solution()
