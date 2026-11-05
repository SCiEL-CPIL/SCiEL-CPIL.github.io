<<<<<<< SEARCH
    def add_4x4_term(a_blocks, b_blocks, c_blocks, col_start):
        """
        Add a 4x4x4 Strassen term using 2x2x2 Strassen sub-factors.
        a_blocks: dict {(br,bc): coeff} - linear combo of A's 2x2 blocks
        b_blocks: dict {(br,bc): coeff} - linear combo of B's 2x2 blocks  
        c_blocks: dict {(br,bc): coeff} - linear combo of C's 2x2 blocks
        """
        for inner_k in range(7):
            col = col_start + inner_k
            u_vec = s_u[:, inner_k]
            v_vec = s_v[:, inner_k]
            w_vec = s_w[:, inner_k]
            
            # Distribute u_vec to A entries
            for (a_br, a_bc), a_coeff in a_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            gi = 2 * a_br + lr
                            gj = 2 * a_bc + lc
                            U[gi * 5 + gj, col] += a_coeff * u_val
            
            # Distribute v_vec to B entries
            for (b_br, b_bc), b_coeff in b_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            gj = 2 * b_br + lr
                            gk = 2 * b_bc + lc
                            V[gj * 5 + gk, col] += b_coeff * v_val
            
            # Distribute w_vec to C entries
            for (c_br, c_bc), c_coeff in c_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            gi = 2 * c_br + lr
                            gk = 2 * c_bc + lc
                            W[gk * 4 + gi, col] += c_coeff * w_val
=======
    def add_4x4_term(a_blocks, b_blocks, c_blocks, col_start):
        """
        Add a 4x4x4 Strassen term using 2x2x2 Strassen sub-factors (vectorized).
        """
        for inner_k in range(7):
            col = col_start + inner_k
            u_vec = s_u[:, inner_k]
            v_vec = s_v[:, inner_k]
            w_vec = s_w[:, inner_k]
            
            # Build 4x4 matrices for A, B, C contributions
            A_mat = np.zeros((4, 4), dtype=int)
            B_mat = np.zeros((4, 4), dtype=int)
            C_mat = np.zeros((4, 4), dtype=int)
            
            for (a_br, a_bc), a_coeff in a_blocks.items():
                A_mat[2*a_br:2*a_br+2, 2*a_bc:2*a_bc+2] += a_coeff * u_vec.reshape(2,2)
            for (b_br, b_bc), b_coeff in b_blocks.items():
                B_mat[2*b_br:2*b_br+2, 2*b_bc:2*b_bc+2] += b_coeff * v_vec.reshape(2,2)
            for (c_br, c_bc), c_coeff in c_blocks.items():
                C_mat[2*c_br:2*c_br+2, 2*c_bc:2*c_bc+2] += c_coeff * w_vec.reshape(2,2)
            
            # Flatten and add to U, V, W columns
            U[:20, col] += A_mat.flatten()
            V[:20, col] += B_mat.flatten()
            W[:16, col] += C_mat.flatten()
>>>>>>> REPLACE

