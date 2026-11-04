def add_rank1_term(U, V, W, col, u_sparse, v_sparse, w_sparse, n, m, p):
    # u_sparse: dict (i, j) -> coeff
    # v_sparse: dict (j, k) -> coeff
    # w_sparse: dict (k, i) -> coeff
    for (i, j), coeff in u_sparse.items():
        U[i*m+j, col] += coeff
    for (j, k), coeff in v_sparse.items():
        V[j*p+k, col] += coeff
    for (k, i), coeff in w_sparse.items():
        W[k*n+i, col] += coeff
