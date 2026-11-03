    # Part 3
    for c in range(16):
        for i in range(16):
            U_final[i][col_offset + c] = U_4x4x1[i][c]
            # V index: j*9 + 8
            j = i // 4
            k = i % 4 # k is 0, but we only have 1 col, so k=0
            # In V_4x4x1, index is j.
            v_idx = j * 9 + 8
            V_final[v_idx][col_offset + c] = V_4x4x1[j][c]
            
            # W index: 8*4 + i
            # In W_4x4x1, index is i.
            w_idx = 8 * 4 + i
            W_final[w_idx][col_offset + c] = W_4x4x1[i][c]
