    def add_block(u_map, v_map, w_map, block_id):
        col_start = block_id * 7
        for r in range(7):
            # Update U
            for a_idx in range(4):
                val = S_A[r][a_idx]
                if val != 0:
                    u_idx = u_map[a_idx]
                    U[u_idx][col_start + r] += val
                    
            # Update V
            for b_idx in range(4):
                val = S_B[r][b_idx]
                if val != 0:
                    v_idx = v_map[b_idx]
                    V[v_idx][col_start + r] += val
                    
            # Update W
            for c_idx in range(4):
                val = S_C[r][c_idx]
                if val != 0:
                    w_idx = w_map[c_idx]
                    W[w_idx][col_start + r] += val
