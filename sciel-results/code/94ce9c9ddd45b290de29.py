<<<<<<< SEARCH
    def search_lower_rank(max_rank_to_try=54, time_limit=None):
        """
        Search for a rank < 55 decomposition.
        Given the disjoint support of the j=0 and j=1,2 parts,
        we focus on the 2x2 x 2x10 part.
        We try a limited random search for rank 34 (total 54) in the 2x2 x 2x10 subspace.
        """
        if time_limit is None:
            time_limit = DEADLINE - 10.0 - (time.time() - start_time)
        
        if time_limit <= 0:
            return None

        # We only search for rank 54 (saving 1 term)
        # This implies rank 34 for the 2x2 x 2x10 part.
        # This is a very hard search space.
        # We will try a few random perturbations of the Strassen solution for the 2x2 x 2x10 part.
        
        # For efficiency, we skip the full random search and just return None,
        # as the probability of finding a rank 34 decomposition for 2x2x2x10
        # with simple random integer/half-integer search is extremely low.
        
        return None
=======
    def search_lower_rank(max_rank_to_try=54, time_limit=None):
        """
        Search for a rank < 55 decomposition using ALS optimization.
        Start from the rank-55 Strassen decomposition and attempt to find
        a valid rank-54 decomposition via alternating least squares.
        """
        if time_limit is None:
            time_limit = DEADLINE - 10.0 - (time.time() - start_time)
        
        if time_limit <= 0:
            return None

        target_rank = 54
        n, m, p = DIMENSIONS
        
        # Build the target tensor T[i,j,k] = 1 for all i,j,k
        # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Start from rank-55 and try to project to rank-54
        base = make_rank_55()
        if base is None:
            return None
        
        U55 = np.array(base["U"], dtype=float)
        V55 = np.array(base["V"], dtype=float)
        W55 = np.array(base["W"], dtype=float)
        
        # Initialize rank-54 by dropping the last column (or a random column)
        # and then optimize
        random.seed(42)
        
        for attempt in range(3):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # Drop last column
                U = U55[:, :target_rank].copy()
                V = V55[:, :target_rank].copy()
                W = W55[:, :target_rank].copy()
            elif attempt == 1:
                # Drop a random column and add small noise
                drop_idx = random.randint(0, 54)
                keep = [i for i in range(55) if i != drop_idx]
                U = U55[:, keep].copy()
                V = V55[:, keep].copy()
                W = W55[:, keep].copy()
                # Add small noise
                U += np.random.randn(*U.shape) * 0.01
                V += np.random.randn(*V.shape) * 0.01
                W += np.random.randn(*W.shape) * 0.01
            else:
                # Use SVD to find best rank-54 approximation
                # Concatenate and do a joint optimization
                U = U55[:, :target_rank].copy()
                V = V55[:, :target_rank].copy()
                W = W55[:, :target_rank].copy()
                # Scale to compensate for missing term
                scale = 55.0 / 54.0
                U *= scale
                V *= scale
                W *= scale
            
            # ALS optimization
            max_iter = 200
            for it in range(max_iter):
                if time.time() - start_time > DEADLINE - 15:
                    break
                
                # Update U: for each row u_idx, solve least squares
                # U[u_idx, c] = sum over (j,k) where i*m+j=u_idx of T[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
                # This is: U[u_idx] = (V_sub @ W_sub.T) where V_sub and W_sub are selected rows
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
                        # For fixed (i,j), we need sum_c U[u_idx,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                        # This is a linear system: for each k, sum_c U[u_idx,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                        # Let's build the system: A @ U[u_idx] = b
                        # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        # b[k] = 1
                        A = np.zeros((p, target_rank))
                        b = np.ones(p)
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve A @ U[u_idx] = b
                        try:
                            U[u_idx, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        except:
                            pass
                
                # Update V: for each row v_idx (j*p+k), solve
                # V[j*p+k, c] such that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                for j in range(m):
                    for k in range(p):
                        v_idx = j * p + k
                        A = np.zeros((n, target_rank))
                        b = np.ones(n)
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            V[v_idx, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        except:
                            pass
                
                # Update W: for each row w_idx (k*n+i), solve
                # W[k*n+i, c] such that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                for k in range(p):
                    for i in range(n):
                        w_idx = k * n + i
                        A = np.zeros((m, target_rank))
                        b = np.ones(m)
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W[w_idx, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        except:
                            pass
                
                # Check convergence
                if it % 10 == 0 or it == max_iter - 1:
                    # Verify
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                total = 0.0
                                for c in range(target_rank):
                                    total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                                err = abs(total - 1.0)
                                if err > max_err:
                                    max_err = err
                    
                    if max_err < 1e-8:
                        # Found a valid decomposition!
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
                    # Early stop if error is increasing
                    if it > 50 and max_err > 1.0:
                        break
        
        return None
>>>>>>> REPLACE

