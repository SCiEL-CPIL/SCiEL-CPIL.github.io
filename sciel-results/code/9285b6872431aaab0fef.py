<<<<<<< SEARCH
        # Schoolbook part (col 6)
        # C_02 (col 6) = A_00 B_02 + A_01 B_12
        # C_12 (col 6) = A_10 B_02 + A_11 B_12
        # B_02: b_row, col 6
        # B_12: b_row+1, col 6
        
        schoolbook_terms = [
            ({(0,0): 1}, {(0,0): 1}, (0,0)), # A00 B02 -> C00
            ({(0,1): 1}, {(1,0): 1}, (0,0)), # A01 B12 -> C00
            ({(1,0): 1}, {(0,0): 1}, (1,0)), # A10 B02 -> C10
            ({(1,1): 1}, {(1,0): 1}, (1,0))  # A11 B12 -> C10
        ]
        
        for t_idx in range(4):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_pos = schoolbook_terms[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            for (li, lj), c_x in x_sub.items():
                i = 2 * a_row_block + li
                j = 2 * a_col_block + lj
                u_idx = i * m + j
                u_coeffs[u_idx] += c_x
            
            for (li, lj), c_y in y_sub.items():
                i = 2 * b_row_block + li
                j = 6 # Col 6
                v_idx = i * p + j
                v_coeffs[v_idx] += c_y
            
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            
            w_coeffs = [0.0] * (p*n)
            i_final = 2 * c_row_block + c_pos[0]
            k_final = 6
            w_idx = k_final * n + i_final
            w_coeffs[w_idx] = 1.0
            
            for i in range(p*n):
                W[i][r] = w_coeffs[i]
=======
        # Combined schoolbook part (col 6) — 2 terms instead of 4
        # C_02 (col 6) = A_00 B_02 + A_01 B_12  → single term
        # C_12 (col 6) = A_10 B_02 + A_11 B_12  → single term
        # B_02: b_row, col 6; B_12: b_row+1, col 6

        combined_terms = [
            # Term 1: C row 0 (within block), col 6
            # U: A[a_row, a_col] and A[a_row, a_col+1]
            # V: B[b_row, 6] and B[b_row+1, 6]
            # W: C[6, c_row]
            ({(0,0): 1, (0,1): 1}, {(0,0): 1, (1,0): 1}, (0,)),
            # Term 2: C row 1 (within block), col 6
            # U: A[a_row+1, a_col] and A[a_row+1, a_col+1]
            # V: B[b_row, 6] and B[b_row+1, 6]
            # W: C[6, c_row+1]
            ({(1,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 1}, (1,)),
        ]

        for t_idx in range(2):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_pos = combined_terms[t_idx]

            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)

            for (li, lj), c_x in x_sub.items():
                i = 2 * a_row_block + li
                j = 2 * a_col_block + lj
                u_idx = i * m + j
                u_coeffs[u_idx] += c_x

            for (li, lj), c_y in y_sub.items():
                i = 2 * b_row_block + li
                j = 6 # Col 6
                v_idx = i * p + j
                v_coeffs[v_idx] += c_y

            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]

            w_coeffs = [0.0] * (p*n)
            i_final = 2 * c_row_block + c_pos[0]
            k_final = 6
            w_idx = k_final * n + i_final
            w_coeffs[w_idx] = 1.0

            for i in range(p*n):
                W[i][r] = w_coeffs[i]
>>>>>>> REPLACE
