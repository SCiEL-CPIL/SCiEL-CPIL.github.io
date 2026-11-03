    col = 0
    for step_idx, (a_blk1, a_sign1, a_blk2, a_sign2, b_blk1, b_sign1, b_blk2, b_sign2) in enumerate(strassen_steps):
        for sub_i in range(s):
            for sub_j in range(s):
                for sub_k in range(s):
                    # U part
                    if a_sign1 != 0:
                        ai, aj = get_block_indices(a_blk1, sub_i, sub_j)
                        U[ai * m + aj, col] = a_sign1
                    if a_sign2 != 0:
                        ai, aj = get_block_indices(a_blk2, sub_i, sub_j)
                        U[ai * m + aj, col] = a_sign2
                    
                    # V part
                    if b_sign1 != 0:
                        bi, bj = get_block_indices(b_blk1, sub_j, sub_k)
                        V[bi * p + bj, col] = b_sign1
                    if b_sign2 != 0:
                        bi, bj = get_block_indices(b_blk2, sub_j, sub_k)
                        V[bi * p + bj, col] = b_sign2
                    
                    # W part: Sum contributions from all output blocks using this M_step
                    for out_blk, contributions in output_contributions.items():
                        for (m_idx, coeff) in contributions:
                            if m_idx == step_idx:
                                ci, ck = get_block_indices(out_blk, sub_i, sub_k)
                                W[ck * n + ci, col] = coeff
                    
                    col += 1
