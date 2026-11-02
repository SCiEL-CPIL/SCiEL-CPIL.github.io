def add_term(U, V, W, rank_col, a_idx, b_idx, c_idx, coeff=1):
    U[a_idx, rank_col] += coeff
    V[b_idx, rank_col] += coeff
    W[c_idx, rank_col] += coeff
