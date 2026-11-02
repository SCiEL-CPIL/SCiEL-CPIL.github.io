    def try_rank_als(rank, max_iters=200, num_restarts=10):
        """Try to find a decomposition of given rank using ALS."""
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                for a in range(n*m):
                    A_mat = np.zeros((rank, rank))
                    b_vec = np.zeros(rank)
                    for (sa, sb, sc) in support:
                        if sa == a:
                            v_w = V[sb, :] * W[sc, :]
                            A_mat += np.outer(v_w, v_w)
                            b_vec += T_val(sa, sb, sc) * v_w
                    try:
                        U[a, :] = np.linalg.solve(A_mat + 1e-6 * np.eye(rank), b_vec)
                    except np.linalg.LinAlgError:
                        pass
                        
                # Update V
                for b in range(m*p):
                    A_mat = np.zeros((rank, rank))
                    b_vec = np.zeros(rank)
                    for (sa, sb, sc) in support:
                        if sb == b:
                            u_w = U[sa, :] * W[sc, :]
                            A_mat += np.outer(u_w, u_w)
                            b_vec += T_val(sa, sb, sc) * u_w
                    try:
                        V[b, :] = np.linalg.solve(A_mat + 1e-6 * np.eye(rank), b_vec)
                    except np.linalg.LinAlgError:
                        pass

                # Update W
                for c in range(p*n):
                    A_mat = np.zeros((rank, rank))
                    b_vec = np.zeros(rank)
                    for (sa, sb, sc) in support:
                        if sc == c:
                            u_v = U[sa, :] * V[sb, :]
                            A_mat += np.outer(u_v, u_v)
                            b_vec += T_val(sa, sb, sc) * u_v
                    try:
                        W[c, :] = np.linalg.solve(A_mat + 1e-6 * np.eye(rank), b_vec)
                    except np.linalg.LinAlgError:
                        pass

            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            err = compute_error(U_r, V_r, W_r)
            if err < 1e-9:
                return U_r, V_r, W_r
                
        return None
