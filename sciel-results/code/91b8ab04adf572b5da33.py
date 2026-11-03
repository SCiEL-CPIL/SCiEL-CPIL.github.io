    def _remove_linearly_dependent_columns(U, V, W, n, m, p, tol=1e-9, deadline=None):
        rank = len(U[0])
        if rank == 0:
            return U, V, W, []
        
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        dim = n * m * p
        M_list = []
        for r in range(rank):
            M_r = np.zeros(dim)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        M_r[i*m*p + j*p + k] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
            M_list.append(M_r)
        
        M_matrix = np.array(M_list).T # shape (dim, rank)
        
        # Compute SVD
        try:
            U_svd, s, Vh = np.linalg.svd(M_matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            return U, V, W, list(range(rank))
        
        # Determine numerical rank
        num_rank = np.sum(s > tol * s[0])
        
        if num_rank == rank:
            return U, V, W, list(range(rank))
        
        # Keep the first num_rank columns (in the SVD basis)
        # This is a heuristic: we keep the directions with largest singular values.
        # We need to map back to original column indices?
        # SVD doesn't directly give original indices.
        # So this method doesn't directly tell us WHICH columns to remove.
        
        # Greedy is better for identifying specific columns.
        # I will stick to greedy but improve it with randomization.
        
        kept = list(range(rank))
        improved = True
        attempts = 0
        max_attempts = 50
        
        while improved and attempts < max_attempts:
            if deadline is not None and time.time() > deadline:
                break
            improved = False
            current_rank = len(kept)
            if current_rank <= 1:
                break
            
            M_kept = np.array([M_list[r] for r in kept])
            
            best_to_remove = -1
            best_residual = tol * 2
            
            # Randomly shuffle the order of checking to avoid bias
            indices = list(range(current_rank))
            np.random.shuffle(indices)
            
            for idx in indices:
                col_to_remove = kept[idx]
                others = [r for i, r in enumerate(kept) if i != idx]
                if len(others) == 0:
                    continue
                
                M_others = np.array([M_list[r] for r in others])
                
                try:
                    x, residuals, _, _ = np.linalg.lstsq(M_others.T, M_list[col_to_remove], rcond=None)
                    reconstructed = M_others.T @ x
                    residual = np.linalg.norm(M_list[col_to_remove] - reconstructed)
                    
                    if residual < best_residual:
                        best_residual = residual
                        best_to_remove = idx
                except np.linalg.LinAlgError:
                    continue
            
            if best_to_remove >= 0:
                kept.pop(best_to_remove)
                improved = True
                attempts += 1
            else:
                break
        
        if len(kept) == rank:
            return U, V, W, kept
        
        new_rank = len(kept)
        new_U = [[0]*new_rank for _ in range(n*m)]
        new_V = [[0]*new_rank for _ in range(m*p)]
        new_W = [[0]*new_rank for _ in range(p*n)]
        
        for new_idx, old_idx in enumerate(kept):
            for i in range(n*m):
                new_U[i][new_idx] = U[i][old_idx]
            for j in range(m*p):
                new_V[j][new_idx] = V[j][old_idx]
            for k in range(p*n):
                new_W[k][new_idx] = W[k][old_idx]
        
        return new_U, new_V, new_W, kept
    