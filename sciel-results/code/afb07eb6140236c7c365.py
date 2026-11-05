    # Remove terms with no W contribution
    clean_terms = {k: v for k, v in terms_dict.items() if v}
    rank = len(clean_terms)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_idx = 0
    for (a_flat, b_flat, coeff), w_dict in clean_terms.items():
        U[a_flat, col_idx] = coeff
        V[b_flat, col_idx] = 1
        for w_flat, w_coeff in w_dict.items():
            W[w_flat, col_idx] = w_coeff
        col_idx += 1
