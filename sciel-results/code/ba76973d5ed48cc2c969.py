<<<<<<< SEARCH
    # Try to find if any term is in the span of others - efficient approach
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5 and rank > 0:
        # Build the full tensor matrix: shape (n*m*p, rank)
        tensor_mat = np.zeros((n*m*p, rank))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_mat[i*m*p + j*p + k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Use SVD to find the rank and identify which columns to keep
        try:
            U_svd, s, Vt_svd = np.linalg.svd(tensor_mat, full_matrices=False)
            # s is the list of singular values
            if len(s) > 0:
                tol = 1e-10 * s[0]
                significant_sv = np.where(s > tol)[0]
                current_rank = len(significant_sv)
                
                if current_rank < rank:
                    # Keep only the first 'current_rank' right singular vectors
                    # These correspond to the most important directions
                    keep_V = Vt_svd[:current_rank].T  # shape (rank, current_rank)
                    # We need to express the original columns in terms of these
                    # The kept columns are a basis for the column space
                    
                    # Actually, simpler: just zero out the dependent columns
                    # Use the fact that if rank is less than rank, some columns are dependent
                    # Find which columns to keep using QR with pivoting on the reduced matrix
                    
                    # Alternative: use the left singular vectors to project
                    # The column space is spanned by U_svd[:, :current_rank]
                    # Project each original column onto this space and check residual
                    
                    # Simpler approach: iteratively remove columns that are in span of others
                    # Start from the end and check
                    for r in range(rank - 1, -1, -1):
                        if time.time() - start_time > DEADLINE:
                            break
                        if r >= tensor_mat.shape[1]:
                            continue
                        
                        other_cols = [c for c in range(tensor_mat.shape[1]) if c != r]
                        if not other_cols:
                            continue
                        
                        # Check if column r is in span of others using least squares
                        others_mat = tensor_mat[:, other_cols]
                        try:
                            coeffs, residuals, rk, sv = np.linalg.lstsq(others_mat, tensor_mat[:, r], rcond=1e-10)
                            reconstruction = others_mat @ coeffs
                            if np.allclose(reconstruction, tensor_mat[:, r], atol=1e-8):
                                U = np.delete(U, r, axis=1)
                                V = np.delete(V, r, axis=1)
                                W = np.delete(W, r, axis=1)
                                tensor_mat = np.delete(tensor_mat, r, axis=1)
                                rank -= 1
                                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                best_rank = rank
                        except:
                            pass
        except:
            pass
=======
    # Try to find if any term is in the span of others - efficient approach
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5 and rank > 0:
        # Build the full tensor matrix: shape (n*m*p, rank)
        tensor_mat = np.zeros((n*m*p, rank))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_mat[i*m*p + j*p + k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Use QR with column pivoting to find a minimal set of linearly independent columns
        try:
            if rank > 0:
                # QR with pivoting: A[:, P] = Q * R
                # The first 'num_independent' columns of P form a basis
                A = tensor_mat.copy()
                m_tensor, n_tensor = A.shape
                
                # Manual QR with column pivoting for better control
                Q = np.zeros((m_tensor, n_tensor))
                R = np.zeros((n_tensor, n_tensor))
                P = list(range(n_tensor))
                
                for k in range(n_tensor):
                    if time.time() - start_time > DEADLINE:
                        break
                    # Find column with largest norm in remaining submatrix
                    if k < m_tensor:
                        col_norms = np.sqrt(np.sum(A[k:, k:]**2, axis=0))
                    else:
                        col_norms = np.zeros(n_tensor - k)
                    
                    if np.max(col_norms) < 1e-12:
                        # All remaining columns are zero
                        for j in range(k, n_tensor):
                            P[j] = j
                        break
                    
                    # Find pivot column
                    pivot_local = np.argmax(col_norms)
                    pivot_global = k + pivot_local
                    
                    # Swap columns k and pivot_global in P
                    P[k], P[pivot_global] = P[pivot_global], P[k]
                    
                    # Extract column k from A using P
                    col = A[:, P[k]]
                    if k < m_tensor:
                        col_sub = col[k:]
                    else:
                        col_sub = col
                    
                    norm = np.linalg.norm(col_sub)
                    if norm < 1e-12:
                        R[k, k] = 0
                        continue
                    
                    # Q column
                    if k < m_tensor:
                        Q[k:, P[k]] = col_sub / norm
                    else:
                        Q[:, P[k]] = col / norm
                    R[k, k] = norm
                    
                    # Eliminate from other columns
                    for j in range(k + 1, n_tensor):
                        if j < m_tensor:
                            R[k, j] = np.dot(Q[k:, P[k]], A[k:, P[j]])
                        else:
                            R[k, j] = np.dot(Q[:, P[k]], A[:, P[j]])
                        if j < m_tensor:
                            A[k:, P[j]] -= R[k, j] * Q[k:, P[k]]
                        else:
                            A[:, P[j]] -= R[k, j] * Q[:, P[k]]
                
                # Determine numerical rank from R diagonal
                diag_R = np.abs(np.diag(R))
                if len(diag_R) > 0:
                    tol = 1e-10 * max(diag_R[0], 1e-15)
                    num_independent = int(np.sum(diag_R > tol))
                else:
                    num_independent = 0
                
                if 0 < num_independent < rank:
                    # Keep only the first 'num_independent' pivot columns
                    keep_indices = [P[i] for i in range(num_independent)]
                    
                    # Verify: check that removing other columns still gives exact decomposition
                    U_new = U[:, keep_indices]
                    V_new = V[:, keep_indices]
                    W_new = W[:, keep_indices]
                    
                    if verify_decomp(U_new, V_new, W_new):
                        U = U_new
                        V = V_new
                        W = W_new
                        rank = num_independent
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = rank
                        tensor_mat = tensor_mat[:, keep_indices]
        except:
            pass
        
        # Second pass: try to remove individual columns that are in span of others
        # This can catch cases where QR pivoting didn't give the optimal subset
        remaining_time = DEADLINE - (time.time() - start_time)
        if remaining_time > 3 and rank > 1:
            # Rebuild tensor_mat if we changed it
            if tensor_mat.shape[1] != rank:
                tensor_mat = np.zeros((n*m*p, rank))
                for r in range(rank):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                tensor_mat[i*m*p + j*p + k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Greedy removal from end
            for r in range(rank - 1, -1, -1):
                if time.time() - start_time > DEADLINE:
                    break
                if r >= tensor_mat.shape[1]:
                    continue
                
                other_cols = [c for c in range(tensor_mat.shape[1]) if c != r]
                if not other_cols:
                    continue
                
                # Check if column r is in span of others using least squares
                others_mat = tensor_mat[:, other_cols]
                try:
                    coeffs, residuals, rk, sv = np.linalg.lstsq(others_mat, tensor_mat[:, r], rcond=1e-10)
                    reconstruction = others_mat @ coeffs
                    if np.allclose(reconstruction, tensor_mat[:, r], atol=1e-8):
                        U = np.delete(U, r, axis=1)
                        V = np.delete(V, r, axis=1)
                        W = np.delete(W, r, axis=1)
                        tensor_mat = np.delete(tensor_mat, r, axis=1)
                        rank -= 1
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = rank
                except:
                    pass
>>>>>>> REPLACE

Now let me also add an improvement: try alternative Strassen block arrangements. The current approach uses Strassen only on A[0:2,0:2]*B[0:2,0:8]. We could also try using Strassen on other 2×2 blocks to potentially get more rank reduction. Let me add a second phase that tries different block decompositions and picks the best.

