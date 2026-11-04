    def add_term(term_idx, u_blocks, v_blocks, w_blocks):
        for i in range(4):
            for j in range(4):
                u_val = 0
                for block_id, sign in u_blocks:
                    br = block_id // 2
                    bc = block_id % 2
                    # Block (br, bc) covers rows 2*br to 2*br+1 and cols 2*bc to 2*bc+1
                    if (2*br <= i <= 2*br+1) and (2*bc <= j <= 2*bc+1):
                        u_val += sign
                if u_val != 0:
                    U[i * 4 + j][term_idx] = u_val
                    
                v_val = 0
                for block_id, sign in v_blocks:
                    br = block_id // 2
                    bc = block_id % 2
                    if (2*br <= i <= 2*br+1) and (2*bc <= j <= 2*bc+1):
                        v_val += sign
                if v_val != 0:
                    V[i * 4 + j][term_idx] = v_val
                    
                w_val = 0
                for block_id, sign in w_blocks:
                    br = block_id // 2
                    bc = block_id % 2
                    if (2*br <= i <= 2*br+1) and (2*bc <= j <= 2*bc+1):
                        w_val += sign
                if w_val != 0:
                    W[i * 4 + j][term_idx] = w_val
