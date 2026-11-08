    def try_reduce_rank(U, V, W, target_rank):
        """Try to find a valid decomposition of target_rank by removing redundant columns."""
        nm, cur_rank = U.shape
        mp, _ = V.shape
        pn, _ = W.shape
        
        # Precompute T_true for efficiency
        T_true = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i * m + j, j * p + k, k * n + i] = 1.0

        # Try each column to remove
        for remove_idx in range(cur_rank):
            if time.time() - start_time > deadline - 10:
                break
            
            # Remove the column
            U_sub = np.delete(U, remove_idx, axis=1)
            V_sub = np.delete(V, remove_idx, axis=1)
            W_sub = np.delete(W, remove_idx, axis=1)
            
            # T_decomp shape: (n*m, m*p, p*n)
            T_decomp = np.einsum('ir,jr,kr->ijk', U_sub, V_sub, W_sub)
            T_error = T_true - T_decomp
            
            # Check if the removed column was redundant (error is zero)
            if np.allclose(T_error, 0, atol=1e-9):
                if cur_rank - 1 == target_rank:
                    new_U = np.zeros((nm, target_rank))
                    new_V = np.zeros((mp, target_rank))
                    new_W = np.zeros((pn, target_rank))
                    new_U[:, :] = U_sub
                    new_V[:, :] = V_sub
                    new_W[:, :] = W_sub
                    if verify(new_U, new_V, new_W):
                        return new_U, new_V, new_W
                continue

            # If error is not zero, the column was not redundant.
            # We cannot simply add a rank-1 term and reduce the total rank.
            # That would keep the rank at cur_rank.
            # So we skip this column.
            pass

        return None
