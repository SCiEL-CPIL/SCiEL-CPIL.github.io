    def try_find_decomposition(rank, num_restarts=100):
        # Use more restarts since objective is faster
        for restart in range(num_restarts):
            if not check_time():
                return None
            
            # Safety margin: stop if less than 10s left
            if time.time() - START_TIME > BUDGET_SECONDS - 10:
                return None

            size = rank * (dim_u + dim_v + dim_w)
            x0 = np.random.randn(size)
            
            try:
                result = least_squares(objective, x0, args=(rank,), method='lm', max_nfev=2000)
            except Exception:
                continue
            
            if result.cost < 1e-8:
                # Extract and round
                U_flat = result.x[:rank * dim_u]
                V_flat = result.x[rank * dim_u : rank * dim_u + rank * dim_v]
                W_flat = result.x[rank * dim_u + rank * dim_v :]
                
                U = U_flat.reshape(rank, dim_u).T
                V = V_flat.reshape(rank, dim_v).T
                W = W_flat.reshape(rank, dim_w).T
                
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify
                valid = True
                # Use the fast verification
                U_3d = U_r.reshape(n, m, rank)
                V_3d = V_r.reshape(m, p, rank)
                W_3d = W_r.reshape(p, n, rank)
                T_computed = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                
                if np.allclose(T_computed, 1.0, atol=1e-9):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        return None
    