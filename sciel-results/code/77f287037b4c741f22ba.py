def _try_global_elimination(U, V, W, deadline):
    rank = U.shape[1]
    K = _build_kronecker_matrix(U, V, W)
    try:
        s = np.linalg.svd(K, compute_uv=False)
    except:
        return U, V, W
        
    tol = 1e-9
    if np.sum(s > tol) < rank:
        # Dependency exists
        # Find null space
        U_svd, s, Vt = np.linalg.svd(K, compute_uv=True)
        null_vecs = Vt[np.sum(s > tol):, :]
        
        # Try to eliminate columns based on null vectors
        # For each null vector, find the column with max abs component
        # and try to remove it.
        
        for nv in null_vecs:
            # nv is a vector of length rank
            # It represents a linear combination of columns that sums to 0 in the tensor space
            # We can use this to eliminate one column.
            # However, we need to update U, V, W.
            # This is complex. 
            # The existing _check_column_redundant does this by solving for the coefficients.
            # So we just need to trigger _check_column_redundant on the "best" candidate.
            
            max_idx = np.argmax(np.abs(nv))
            redundant, _ = _check_column_redundant(U, V, W, max_idx)
            if redundant:
                mask = [i for i in range(rank) if i != max_idx]
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                if _verify_decomposition(U_new, V_new, W_new):
                    return U_new, V_new, W_new
    return U, V, W
