<<<<<<< SEARCH
    rank = len(columns)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx][c] = val
        for idx, val in v_d.items():
            V[idx][c] = val
        for idx, val in w_d.items():
            W[idx][c] = val
            
    return {"U": U, "V": V, "W": W}
=======
    rank = len(columns)
    
    # Build dense matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx, c] = val
        for idx, val in v_d.items():
            V[idx, c] = val
        for idx, val in w_d.items():
            W[idx, c] = val
    
    # Check for linearly dependent columns in the tensor space
    # Form Kronecker products and check rank
    if _deadline_exceeded():
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    kron_matrix = np.zeros((n * m * p, rank))
    for c in range(rank):
        kron_matrix[:, c] = np.kron(np.kron(U[:, c], V[:, c]), W[:, c])
    
    # Greedy column selection: keep column if it increases the rank
    kept = []
    current_rank = 0
    for c in range(rank):
        if current_rank == 0:
            kept.append(c)
            current_rank = 1
            continue
        # Check if column c is in the span of previously kept columns
        # Use least squares to check
        sub_matrix = kron_matrix[:, kept]
        # Compute the residual
        try:
            coeffs, residuals, rank_sub, _ = np.linalg.lstsq(sub_matrix, kron_matrix[:, c], rcond=None)
            residual = np.linalg.norm(kron_matrix[:, c] - sub_matrix @ coeffs)
            if residual > 1e-10:
                kept.append(c)
                current_rank += 1
        except np.linalg.LinAlgError:
            kept.append(c)
            current_rank += 1
    
    if len(kept) < rank:
        U = U[:, kept]
        V = V[:, kept]
        W = W[:, kept]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
