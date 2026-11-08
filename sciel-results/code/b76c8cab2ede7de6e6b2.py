    def add_laderman_block(v_k_offset, w_k_offset, start_col):
        # v_k_offset: 0 for block 1, 3 for block 2
        # w_k_offset: 0 for block 1, 3 for block 2
        
        # Term 1: a00(b00+b11-b22)
        # a00: U[0]
        # b00: V[0*7 + 0 + v_k_offset]
        # b11: V[1*7 + 1 + v_k_offset]
        # b22: V[2*7 + 2 + v_k_offset]
        # c00: W[0*3 + 0 + w_k_offset]
        # c11: W[1*3 + 1 + w_k_offset]
        # c22: W[2*3 + 2 + w_k_offset]
        
        add_term(0, 0*7 + 0 + v_k_offset, 0*3 + 0 + w_k_offset, start_col)
        add_term(0, 1*7 + 1 + v_k_offset, 1*3 + 1 + w_k_offset, start_col)
        add_term(0, 2*7 + 2 + v_k_offset, 2*3 + 2 + w_k_offset, start_col, val_v=-1, val_w=-1)
        # ... and so on
