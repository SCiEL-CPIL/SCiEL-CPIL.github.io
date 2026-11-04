<<<<<<< SEARCH
    def try_rank(rank, num_trials=50):
        """Try to find a rank-r decomposition using multiple optimization strategies."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Different initialization strategies
            strategy = trial % 5
            
            if strategy == 0:
                # Pure random
                np.random.seed(1000 + trial * 7 + rank)
                U_init = np.random.randn(u_dim, rank)
                V_init = np.random.randn(v_dim, rank)
                W_init = np.random.randn(w_dim, rank)
            elif strategy == 1:
                # Sparse random (many zeros)
                np.random.seed(2000 + trial * 11 + rank)
                U_init = np.random.randn(u_dim, rank) * (np.random.rand(u_dim, rank) < 0.3)
                V_init = np.random.randn(v_dim, rank) * (np.random.rand(v_dim, rank) < 0.3)
                W_init = np.random.randn(w_dim, rank) * (np.random.rand(w_dim, rank) < 0.3)
            elif strategy == 2:
                # Based on partial schoolbook structure
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                for c in range(min(rank, 18)):
                    i_a = c // (m * p)
                    j_a = (c // p) % m
                    k_b = c % p
                    if i_a < n and j_a < m and k_b < p:
                        U_init[i_a * m + j_a, c] = 1.0
                        V_init[j_a * p + k_b, c] = 1.0
                        W_init[k_b * n + i_a, c] = 1.0
                # Add random perturbation
                np.random.seed(3000 + trial * 13 + rank)
                U_init += np.random.randn(u_dim, rank) * 0.1
                V_init += np.random.randn(v_dim, rank) * 0.1
                W_init += np.random.randn(w_dim, rank) * 0.1
            elif strategy == 3:
                # Block-structured: group terms by j_a
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                # Assign rank terms to different j_a groups
                terms_per_j = max(1, rank // m)
                for j_a in range(m):
                    for t in range(terms_per_j):
                        c = j_a * terms_per_j + t
                        if c >= rank:
                            break
                        # Random within the block
                        np.random.seed(4000 + j_a * 17 + t * 23 + rank)
                        rows_U = np.random.choice(u_dim, size=min(3, u_dim), replace=False)
                        rows_V = np.random.choice(v_dim, size=min(3, v_dim), replace=False)
                        rows_W = np.random.choice(w_dim, size=min(3, w_dim), replace=False)
                        for r in rows_U:
                            U_init[r, c] = np.random.choice([-1, 0, 1])
                        for r in rows_V:
                            V_init[r, c] = np.random.choice([-1, 0, 1])
                        for r in rows_W:
                            W_init[r, c] = np.random.choice([-1, 0, 1])
            else:
                # Winograd-like: try to exploit the structure
                # For 2x3x3, think of it as two 1x3x3 multiplications
                # Each 1x3x3 can potentially use fewer than 3 multiplications
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                
                # Try a structured approach where we combine terms
                # Group by output row i_c
                for i_c in range(n):
                    for t in range(rank // n):
                        c = i_c * (rank // n) + t
                        if c >= rank:
                            break
                        np.random.seed(5000 + i_c * 31 + t * 41 + rank)
                        # A factors: rows i_c*m + j_a for j_a in [0,m)
                        for j_a in range(m):
                            U_init[i_c * m + j_a, c] = np.random.choice([-1, 0, 1])
                        # B factors: cols j_a*p + k_b
                        for j_a in range(m):
                            for k_b in range(p):
                                V_init[j_a * p + k_b, c] = np.random.choice([-1, 0, 1]) * 0.5
                        # C factors: rows k_b*n + i_c
                        for k_b in range(p):
                            W_init[k_b * n + i_c, c] = np.random.choice([-1, 0, 1])
            
            # Flatten parameters
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(params):
                U_r = params[:u_dim * rank].reshape(u_dim, rank)
                V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                diff = recon.flatten() - T_flat
                return np.dot(diff, diff)
            
            # Use multiple optimization methods
            if trial % 3 == 0:
                method = 'L-BFGS-B'
                options = {'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            elif trial % 3 == 1:
                method = 'Nelder-Mead'
                options = {'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            else:
                method = 'Powell'
                options = {'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-10}
            
            try:
                result = minimize(objective, params_init, method=method, options=options)
            except Exception:
                continue
            
            if result.fun < 1e-10:
                params = result.x
                U_r = params[:u_dim * rank].reshape(u_dim, rank)
                V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                
                if check_exact(U_r, V_r, W_r, tol=1e-6):
                    return U_r, V_r, W_r
                
                # Try to polish and round
                if result.fun < 1e-6:
                    # Attempt rounding to half-integers
                    U_round = np.round(U_r * 2) / 2
                    V_round = np.round(V_r * 2) / 2
                    W_round = np.round(W_r * 2) / 2
                    if check_exact(U_round, V_round, W_round, tol=1e-6):
                        return U_round, V_round, W_round
        
        return None
=======
    def try_rank_als(rank, num_trials=100):
        """Try to find a rank-r decomposition using Alternating Least Squares (ALS)."""
        T_u = T.reshape(u_dim, v_dim * w_dim)
        T_v = T.reshape(v_dim, u_dim * w_dim)
        T_w = T.reshape(w_dim, u_dim * v_dim)
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Initialization: Random Gaussian
            np.random.seed(42 + trial * 100 + rank)
            U = np.random.randn(u_dim, rank)
            V = np.random.randn(v_dim, rank)
            W = np.random.randn(w_dim, rank)
            
            # ALS iterations
            for _ in range(50):
                # Solve for U
                # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
                # Reshape T to (u_dim, v_dim * w_dim)
                # Z_u: (v_dim * w_dim) x rank
                Z_u = np.zeros((v_dim * w_dim, rank))
                for c in range(rank):
                    Z_u[:, c] = np.outer(V[:, c], W[:, c]).flatten()
                
                # Solve U * Z_u^T = T_u  =>  U = T_u * (Z_u^T Z_u)^{-1} Z_u^T
                # Using lstsq is safer
                U_new, _, _, _ = np.linalg.lstsq(Z_u.T, T_u.T, rcond=None)
                U = U_new.T
                
                # Solve for V
                Z_v = np.zeros((u_dim * w_dim, rank))
                for c in range(rank):
                    Z_v[:, c] = np.outer(U[:, c], W[:, c]).flatten()
                V_new, _, _, _ = np.linalg.lstsq(Z_v.T, T_v.T, rcond=None)
                V = V_new.T
                
                # Solve for W
                Z_w = np.zeros((u_dim * v_dim, rank))
                for c in range(rank):
                    Z_w[:, c] = np.outer(U[:, c], V[:, c]).flatten()
                W_new, _, _, _ = np.linalg.lstsq(Z_w.T, T_w.T, rcond=None)
                W = W_new.T
                
                # Check convergence / exactness
                if check_exact(U, V, W, tol=1e-8):
                    # Try to snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    if check_exact(U_snap, V_snap, W_snap, tol=1e-6):
                        return U_snap, V_snap, W_snap
                    else:
                        return U, V, W
                        
        return None
>>>>>>> REPLACE

