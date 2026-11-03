<<<<<<< SEARCH
    # Part 2
    for c in range(49):
        for i in range(16):
            U_final[i][col_offset + c] = U_4x4x4[i][c]
            j = i // 4
            k = i % 4
            v_idx = j * 9 + (k + 4)
            V_final[v_idx][col_offset + c] = V_4x4x4[i][c]
            
            k_w = i // 4
            i_w = i % 4
            w_idx = (k_w + 4) * 4 + i_w
            W_final[w_idx][col_offset + c] = W_4x4x4[i][c]
    col_offset += 49
=======
    # Part 2
    for c in range(49):
        for i in range(16):
            U_final[i][col_offset + c] = U_4x4x4[i][c]
            # V index: local B index i (row j, col k) -> global B index (row j, col k+4)
            j = i // 4
            k = i % 4
            v_idx = j * 9 + (k + 4)
            V_final[v_idx][col_offset + c] = V_4x4x4[i][c]
            
            # W index: local C index i (row r, col c) -> global C index (row r, col c+4)
            # W global index is col_global * 4 + row_global
            w_row = i // 4
            w_col = i % 4
            w_idx = (w_col + 4) * 4 + w_row
            W_final[w_idx][col_offset + c] = W_4x4x4[i][c]
    col_offset += 49
>>>>>>> REPLACE
