def decompose_flex(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols):
    """
    Flexible decomposition trying all split points.
    """
    if time.time() > deadline:
        return
    
    if nc == 1 and mc == 1 and pc == 1:
        u_idx = ii * m_dim + jj
        v_idx = jj * p_dim + kk
        w_idx = kk * n_dim + ii
        cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
        return
    
    if nc == 1 or mc == 1 or pc == 1:
        direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
        return
    
    # Try all possible splits
    best = None
    best_rank = float('inf')
    
    # Try splitting nc
    for split in range(1, nc):
        nc1, nc2 = split, nc - split
        if nc1 >= 2 and nc2 >= 2 and mc >= 2 and pc >= 2:
            # Both parts can use Strassen
            cols_temp = []
            decompose_flex(nc1, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols_temp)
            decompose_flex(nc2, mc, pc, ii + nc1, jj, kk, n_dim, m_dim, p_dim, cols_temp)
            rank = len(cols_temp)
            if rank < best_rank:
                best_rank = rank
                best = cols_temp[:]
        else:
            cols_temp = []
            decompose_flex(nc1, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols_temp)
            decompose_flex(nc2, mc, pc, ii + nc1, jj, kk, n_dim, m_dim, p_dim, cols_temp)
            rank = len(cols_temp)
            if rank < best_rank:
                best_rank = rank
                best = cols_temp[:]
    
    # Try splitting mc
    for split in range(1, mc):
        mc1, mc2 = split, mc - split
        cols_temp = []
        decompose_flex(nc, mc1, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols_temp)
        decompose_flex(nc, mc2, pc, ii, jj + mc1, kk, n_dim, m_dim, p_dim, cols_temp)
        rank = len(cols_temp)
        if rank < best_rank:
            best_rank = rank
            best = cols_temp[:]
    
    # Try splitting pc
    for split in range(1, pc):
        pc1, pc2 = split, pc - split
        cols_temp = []
        decompose_flex(nc, mc, pc1, ii, jj, kk, n_dim, m_dim, p_dim, cols_temp)
        decompose_flex(nc, mc, pc2, ii, jj, kk + pc1, n_dim, m_dim, p_dim, cols_temp)
        rank = len(cols_temp)
        if rank < best_rank:
            best_rank = rank
            best = cols_temp[:]
    
    # Also try Strassen if applicable
    if nc >= 2 and mc >= 2 and pc >= 2:
        cols_temp = []
        cols_temp.extend(strassen_2x2x2_cols(ii, jj, kk, n_dim, m_dim, p_dim))
        if nc > 2:
            decompose_flex(nc - 2, mc, pc, ii + 2, jj, kk, n_dim, m_dim, p_dim, cols_temp)
        if mc > 2:
            decompose_flex(2, mc - 2, pc, ii, jj + 2, kk, n_dim, m_dim, p_dim, cols_temp)
        if pc > 2:
            decompose_flex(2, 2, pc - 2, ii, jj, kk + 2, n_dim, m_dim, p_dim, cols_temp)
        rank = len(cols_temp)
        if rank < best_rank:
            best_rank = rank
            best = cols_temp[:]
    
    if best is not None:
        cols.extend(best)
    else:
        direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
