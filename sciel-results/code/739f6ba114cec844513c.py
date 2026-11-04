<<<<<<< SEARCH
def _search_rank_27(U_arr, V_arr, W_arr, T, deadline):
    """Search for rank-27 by checking if any term is linearly dependent."""
    rank = 28
    nm, mp, pn = T.shape

    # Build M matrix: M[abc, r] = U[a,r]*V[b,r]*W[c,r]
    M = np.zeros((nm * mp * pn, rank), dtype=float)
    for r in range(rank):
        M[:, r] = np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r]).flatten()

    # For each candidate term to eliminate, check if it's in the span of the others
    for r_elim in range(rank):
        if time.time() > deadline - 5:
            break

        other_cols = [s for s in range(rank) if s != r_elim]
        M_sub = M[:, other_cols]  # 1024 x 27
        rhs = M[:, r_elim]  # 1024

        # Solve least squares
        try:
            alpha, residuals, rk, sv = np.linalg.lstsq(M_sub, rhs, rcond=1e-10)
            residual = np.max(np.abs(M_sub @ alpha - rhs))
            if residual < 1e-8:
                # Term r_elim is redundant!
                new_U = np.delete(U_arr, r_elim, axis=1)
                new_V = np.delete(V_arr, r_elim, axis=1)
                new_W = np.delete(W_arr, r_elim, axis=1)
                return new_U, new_V, new_W
        except np.linalg.LinAlgError:
            continue

    # Try a different approach: exploit block symmetry
    # The tensor has symmetry between the two rows of A and the two columns of C
    # Try to find a decomposition that respects this symmetry more explicitly

    # Alternative: try to merge terms across blocks
    # For blocks 1 and 2 (both contribute to C columns 0-1), see if terms can be shared
    # This requires a more sophisticated search

    return None
=======
def _search_rank_27(U_arr, V_arr, W_arr, T, deadline):
    """Search for rank-27 by checking linear dependencies and cross-block merging."""
    rank = 28
    nm, mp, pn = T.shape

    # Build M matrix: M[abc, r] = U[a,r]*V[b,r]*W[c,r]
    M = np.zeros((nm * mp * pn, rank), dtype=float)
    for r in range(rank):
        M[:, r] = np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r]).flatten()

    # Check overall rank of M
    try:
        sv = np.linalg.svd(M, full_matrices=False)
        effective_rank = np.sum(sv > 1e-8)
        if effective_rank < rank:
            # Find a basis of size effective_rank
            # Use QR with column pivoting to find independent columns
            Q, R, piv = np.linalg.qr(M.T, mode='complete', pivoting=True)
            # R is upper triangular; find first effective_rank columns
            U_new = U_arr[:, piv[:effective_rank]]
            V_new = V_arr[:, piv[:effective_rank]]
            W_new = W_arr[:, piv[:effective_rank]]
            return U_new, V_new, W_new
    except np.linalg.LinAlgError:
        pass

    # For each candidate term to eliminate, check if it's in the span of the others
    for r_elim in range(rank):
        if time.time() > deadline - 10:
            break

        other_cols = [s for s in range(rank) if s != r_elim]
        M_sub = M[:, other_cols]  # 1024 x 27
        rhs = M[:, r_elim]  # 1024

        # Solve least squares
        try:
            alpha, residuals, rk, sv = np.linalg.lstsq(M_sub, rhs, rcond=1e-10)
            residual = np.max(np.abs(M_sub @ alpha - rhs))
            if residual < 1e-8:
                # Term r_elim is redundant!
                new_U = np.delete(U_arr, r_elim, axis=1)
                new_V = np.delete(V_arr, r_elim, axis=1)
                new_W = np.delete(W_arr, r_elim, axis=1)
                return new_U, new_V, new_W
        except np.linalg.LinAlgError:
            continue

    # Try to find a rank-27 decomposition by allowing linear combinations of terms
    # This is a harder problem; try a few random linear combinations
    if time.time() < deadline - 15:
        rng = np.random.RandomState(42)
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            # Try to eliminate one term by expressing it as a combination of two others
            for r_elim in range(rank):
                # Pick two other terms
                others = [s for s in range(rank) if s != r_elim]
                if len(others) < 2:
                    continue
                r1, r2 = others[trial % len(others)], others[(trial + 1) % len(others)]
                # Check if M[:, r_elim] ≈ a*M[:, r1] + b*M[:, r2]
                A2 = M[:, [r1, r2]]
                rhs = M[:, r_elim]
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A2, rhs, rcond=1e-10)
                    resid = np.max(np.abs(A2 @ coeffs - rhs))
                    if resid < 1e-8:
                        # We can eliminate r_elim
                        # But we need to adjust the other terms
                        # New term r1' = r1 - coeffs[0]*r_elim (but r_elim is gone)
                        # Actually, if r_elim = a*r1 + b*r2, then we just drop r_elim
                        new_U = np.delete(U_arr, r_elim, axis=1)
                        new_V = np.delete(V_arr, r_elim, axis=1)
                        new_W = np.delete(W_arr, r_elim, axis=1)
                        return new_U, new_V, new_W
                except np.linalg.LinAlgError:
                    continue

    return None
>>>>>>> REPLACE

