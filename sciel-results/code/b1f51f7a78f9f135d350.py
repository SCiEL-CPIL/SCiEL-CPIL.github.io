    # Check for linear dependencies in the rank-28 solution
    # Flatten the 28 terms
    terms = []
    for r in range(rank_known):
        u = np.array(U_known[:, r], dtype=float)
        v = np.array(V_known[:, r], dtype=float)
        w = np.array(W_known[:, r], dtype=float)
        term = np.einsum('i,j,k->ijk', u, v, w)
        terms.append(term.flatten())
    
    terms_matrix = np.array(terms)
    # Compute rank
    rank_terms = np.linalg.matrix_rank(terms_matrix, tol=1e-9)
    
    if rank_terms < rank_known:
        # Dependency found. We need to resolve it.
        # For now, just note it. A full resolution is complex.
        # But if rank is 27, we might be able to construct a rank 27 solution.
        pass
