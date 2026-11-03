# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Try to find a decomposition with rank < schoolbook
    # We will test ranks from rank_schoolbook - 1 down to a lower bound.
    # To save time, we only try a few ranks close to schoolbook.
    
    def fast_als_attempt(R, max_iters=30, tol=1e-8, seed=0):
        np.random.seed(seed)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        for it in range(max_iters):
            # Update U_r
            for i in range(n):
                for j in range(m):
                    A_ij = V_r[j] * W_r[:, i]
                    u_sol, _, _, _ = np.linalg.lstsq(A_ij, np.ones(p), rcond=None)
                    U_r[i, j] = u_sol
            
            # Update V_r
            for j in range(m):
                for k in range(p):
                    B_jk = U_r[:, j] * W_r[k]
                    v_sol, _, _, _ = np.linalg.lstsq(B_jk, np.ones(n), rcond=None)
                    V_r[j, k] = v_sol
            
            # Update W_r
            for k in range(p):
                for i in range(n):
                    C_ki = U_r[i] * V_r[:, k]
                    w_sol, _, _, _ = np.linalg.lstsq(C_ki, np.ones(m), rcond=None)
                    W_r[k, i] = w_sol
            
            # Check residual
            T_comp = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            residual = np.linalg.norm(T_comp - 1.0)
            
            if residual < tol:
                return U.copy(), V.copy(), W.copy()
                
        return None

    # Try ranks 83, 82, 81
    for R in range(rank_schoolbook - 1, max(rank_schoolbook - 4, 70), -1):
        for seed in range(3):
            res = fast_als_attempt(R, max_iters=20, seed=seed)
            if res is not None:
                U, V, W = res
                if _verify(U, V, W):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    
    # Fallback to schoolbook
    rank = rank_schoolbook
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
