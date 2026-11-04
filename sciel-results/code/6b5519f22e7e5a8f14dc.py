def _search_2x2x4_rank(r, max_attempts=50, seed=42):
    rng = np.random.RandomState(seed)
    n, m, p = 2, 2, 4
    
    for attempt in range(max_attempts):
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        
        W = np.zeros((n*p, r))
        feasible = True
        
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                # Solve: sum_c W[w_idx,c] * U[i*m+j, c] * V[j*p+k, c] = 1 for j=0,1
                A_sub = np.zeros((m, r))
                for j in range(m):
                    A_sub[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b_sub = np.ones(m)
                
                # Check if solvable
                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                except:
                    feasible = False
                    break
                
                # Check residual
                if residuals.size > 0 and residuals[0] > 1e-10:
                    # Not exactly solvable, but for underdetermined systems,
                    # lstsq gives minimum-norm solution
                    # Check if it actually satisfies
                    if not np.allclose(A_sub @ sol, b_sub, atol=1e-10):
                        feasible = False
                        break
                
                W[w_idx, :] = sol
        
        if not feasible:
            continue
        
        # Verify
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        if np.allclose(T, 1.0, atol=1e-8):
            return U, V, W
    
    return None
