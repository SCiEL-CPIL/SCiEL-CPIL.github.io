                for dk in range(3):
                    for di in range(2):
                        w_row = (k_cb*3 + dk) * n + (i_rb*2 + di)  # C[k, i] -> W[k*n+i]
                        w_block_row = dk * 2 + di  # 0..5
                        for c in range(block_rank):
                            W[w_row][col_offset + c] = W_block[w_block_row][c]
