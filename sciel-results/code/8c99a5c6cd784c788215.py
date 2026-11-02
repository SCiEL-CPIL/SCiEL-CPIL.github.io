    def add_strassen(U, V, W, rank_offset, 
                     a_row_offset, a_col_offset, 
                     b_row_offset, b_col_offset, 
                     c_row_offset, c_col_offset,
                     n_global, m_global, p_global):
        # a is 2x2, b is 2x2, c is 2x2
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # Local indices 0,1 for rows/cols of the 2x2 blocks
        # Global A index: (a_row_offset + i) * m_global + (a_col_offset + j)
        # Global B index: (b_row_offset + j) * p_global + (b_col_offset + k)
        # Global W index: (c_col_offset + k) * n_global + (c_row_offset + i)
        
        # Strassen M1..M7
        # M1 = (x00+x11)(y00+y11)
        # ...
        
        # We will define the coefficients for U, V, W for each of the 7 multiplications.
        # For each multiplication l in 0..6:
        # u_l = sum of x terms
        # v_l = sum of y terms
        # w_l = sum of z terms (coefficients)
        
        # Actually, U, V, W are matrices.
        # For a specific rank column r, U[:, r] is a vector.
        # We can just fill in the non-zero elements.
        
        # Let's define the 7 terms as (u_vec, v_vec, w_vec)
        # where u_vec is a dict {global_idx: coeff}, etc.
        
        terms = []
        
        # M1
        # x: (0,0), (1,1) -> coeff 1
        # y: (0,0), (1,1) -> coeff 1
        # z: (0,0) 1, (1,1) 1, (0,1) -1, (1,0) 1 ? 
        # z00 = M1 + M4 - M5 + M7
        # z11 = M1 - M2 + M3 + M6
        # So M1 contributes +1 to z00, +1 to z11.
        
        # It is easier to define the U, V, W vectors for each M.
        
        # M1
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1} # z00, z11
        
        # M2 = (x10+x11)y00
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(1,0): 1, (1,1): -1} # z10 +1, z11 -1
        
        # M3 = x00(y10-y11)
        u3 = {(0,0): 1}
        v3 = {(1,0): 1, (1,1): -1}
        w3 = {(0,1): 1, (1,1): 1} # z01 +1, z11 +1
        
        # M4 = x11(y01-y00)
        u4 = {(1,1): 1}
        v4 = {(0,1): 1, (0,0): -1}
        w4 = {(0,0): 1, (1,0): 1} # z00 +1, z10 +1
        
        # M5 = (x00+x01)y11
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (0,1): 1} # z00 -1, z01 +1
        
        # M6 = (x10-x00)(y00+y10)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (1,0): 1}
        w6 = {(1,1): 1} # z11 +1
        
        # M7 = (x01-x11)(y01+y11)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(0,1): 1, (1,1): 1}
        w7 = {(0,0): 1, (0,1): -1} # z00 +1, z01 -1
        
        # Check z00: M1(1) + M4(1) - M5(1) + M7(1) = 1+1-1+1 = 2? 
        # Wait. z00 = x00y00 + x01y10.
        # M1 + M4 - M5 + M7
        # (x00+x11)(y00+y11) + x11(y01-y00) - (x00+x01)y11 + (x01-x11)(y01+y11)
        # = x00y00 + x00y11 + x11y00 + x11y11 + x11y01 - x11y00 - x00y11 - x01y11 + x01y01 + x01y11 - x11y01 - x11y11
        # = x00y00 + x01y01 + (x00y11-x00y11) + (x11y00-x11y00) + (x11y11-x11y11) + (x11y01-x11y01) + (-x01y11+x01y11)
        # = x00y00 + x01y01. Correct.
        
        # I will hardcode the U,V,W vectors for these 7 terms.
    