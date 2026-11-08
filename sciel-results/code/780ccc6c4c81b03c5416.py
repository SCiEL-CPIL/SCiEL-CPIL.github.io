    def add_strassen_4x4(offset_u, offset_v, offset_w):
        # offset_u is 0
        # offset_v is the starting column index in B (0 or 4)
        # offset_w is the starting column index in C (0 or 4)
        for k in range(7):
            outer = outer_coeffs[k]
            for l in range(7):
                inner = strassen_coeffs[l]
                col = offset_u + k * 7 + l
                
                # ... U coeffs ...
                
                # V coeffs
                # B_block_IJ is B[2I:2I+2, offset_v + 2J : offset_v + 2J + 2]
                # So B[2I+u, offset_v + 2J+v]
                v_coeffs = {}
                for I, J, c_b in outer['b']:
                    for u, v, c_y in inner['y']:
                        i = 2 * I + u
                        k_b = offset_v + 2 * J + v
                        v_coeffs[(i, k_b)] = v_coeffs.get((i, k_b), 0) + c_b * c_y
                        
                for (i, k_b), val in v_coeffs.items():
                    V[i * p + k_b][col] += val
                    
                # W coeffs
                # C_block_IJ is C[2I:2I+2, offset_w + 2J : offset_w + 2J + 2]
                # C row i, col k_c.
                # W index k_c * n + i.
                w_coeffs = {}
                for I, J, c_c in outer['c']:
                    for u, v, c_z in inner['z']:
                        i = 2 * I + u
                        k_c = offset_w + 2 * J + v
                        w_coeffs[(k_c, i)] = w_coeffs.get((k_c, i), 0) + c_c * c_z
                        
                for (k_c, i), val in w_coeffs.items():
                    W[k_c * n + i][col] += val
