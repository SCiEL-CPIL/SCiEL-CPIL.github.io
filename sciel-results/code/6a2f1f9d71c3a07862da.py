<<<<<<< SEARCH
    def try_als_for_rank(target_rank, max_iters=80, n_restarts=15):
        if time.time() > deadline:
            return None
        
        T_flat_U = T.reshape(n * m, -1)
        T_flat_V = T.reshape(m * p, -1)
        T_flat_W = T.reshape(p * n, -1)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + restart)
            if restart % 3 == 0:
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
            elif restart % 3 == 1:
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
                mask = np.random.random((n * m, target_rank)) > 0.7
                U[mask] = 0
                mask = np.random.random((m * p, target_rank)) > 0.7
                V[mask] = 0
                mask = np.random.random((p * n, target_rank)) > 0.7
                W[mask] = 0
            else:
                U = np.random.randint(0, 2, (n * m, target_rank)) - 0.5
                V = np.random.randint(0, 2, (m * p, target_rank)) - 0.5
                W = np.random.randint(0, 2, (p * n, target_rank)) - 0.5
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                B_mat_T = np.zeros((m * p * p * n, target_rank))
                for r in range(target_rank):
                    B_mat_T[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                try:
                    P_U = np.linalg.pinv(B_mat_T)
                    U = T_flat_U @ P_U
                except:
                    break
                
                B_mat_T_V = np.zeros((n * m * p * n, target_rank))
                for r in range(target_rank):
                    B_mat_T_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                try:
                    P_V = np.linalg.pinv(B_mat_T_V)
                    V = T_flat_V @ P_V
                except:
                    break
                
                B_mat_T_W = np.zeros((n * m * m * p, target_rank))
                for r in range(target_rank):
                    B_mat_T_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                try:
                    P_W = np.linalg.pinv(B_mat_T_W)
                    W = T_flat_W @ P_W
                except:
                    break
                
                if verify(U, V, W):
                    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
                
                if it % 20 == 19:
                    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.max(np.abs(T_approx - T))
                    if err < 1e-4:
                        polished = polish_solution(U, V, W)
                        if polished:
                            return polished
                    
        return None
=======
    def try_als_for_rank(target_rank, max_iters=80, n_restarts=15):
        if time.time() > deadline:
            return None
        
        T_flat_U = T.reshape(n * m, -1)
        T_flat_V = T.reshape(m * p, -1)
        T_flat_W = T.reshape(p * n, -1)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + restart)
            if restart % 3 == 0:
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
            elif restart % 3 == 1:
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
                mask = np.random.random((n * m, target_rank)) > 0.7
                U[mask] = 0
                mask = np.random.random((m * p, target_rank)) > 0.7
                V[mask] = 0
                mask = np.random.random((p * n, target_rank)) > 0.7
                W[mask] = 0
            else:
                U = np.random.randint(0, 2, (n * m, target_rank)) - 0.5
                V = np.random.randint(0, 2, (m * p, target_rank)) - 0.5
                W = np.random.randint(0, 2, (p * n, target_rank)) - 0.5
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                B_mat_T = np.zeros((m * p * p * n, target_rank))
                for r in range(target_rank):
                    B_mat_T[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                try:
                    P_U = np.linalg.pinv(B_mat_T)
                    U = T_flat_U @ P_U
                except:
                    break
                
                B_mat_T_V = np.zeros((n * m * p * n, target_rank))
                for r in range(target_rank):
                    B_mat_T_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                try:
                    P_V = np.linalg.pinv(B_mat_T_V)
                    V = T_flat_V @ P_V
                except:
                    break
                
                B_mat_T_W = np.zeros((n * m * m * p, target_rank))
                for r in range(target_rank):
                    B_mat_T_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                try:
                    P_W = np.linalg.pinv(B_mat_T_W)
                    W = T_flat_W @ P_W
                except:
                    break
                
                if verify_fast(U, V, W, atol=1e-7):
                    polished = polish_solution(U, V, W)
                    if polished:
                        return polished
                    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
                
                if it % 20 == 19:
                    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.max(np.abs(T_approx - T))
                    if err < 1e-4:
                        polished = polish_solution(U, V, W)
                        if polished:
                            return polished
                    
        return None
>>>>>>> REPLACE

