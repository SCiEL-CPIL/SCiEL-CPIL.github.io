def add_block(U, V, W, rank_offset, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
    """
    Adds terms for C[C_rows, C_cols] += A[A_rows, A_cols] * B[B_rows, B_cols]
    where A_cols == B_rows (the summed dimension).
    """
    n_a = len(A_rows)
    m_sum = len(A_cols)
    p_b = len(B_cols)
    
    # The rank of this block is n_a * m_sum * p_b
    # We iterate over all combinations
    for i in A_rows:
        for k in A_cols:
            for j in B_cols:
                # U index: i, k (flattened)
                # V index: k, j (flattened)
                # W index: j, i (flattened)
                
                # Note: The global indices are 0..4
                # U is (25, rank). Index is i*5 + k? 
                # The task says: U[i*m+j, r] where m=5.
                # So U index is i*5 + k.
                # V index is k*5 + j.
                # W index is j*5 + i.
                
                U[i * 5 + k][rank_offset] = 1
                V[k * 5 + j][rank_offset] = 1
                W[j * 5 + i][rank_offset] = 1
                rank_offset += 1
    return rank_offset
