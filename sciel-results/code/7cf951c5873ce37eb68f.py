    def try_reduce_rank(U, V, W, target_rank):
        nm, cur_rank = U.shape
        mp, _ = V.shape
        pn, _ = W.shape
        
        T_true = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i * m + j, j * p + k, k * n + i] = 1.0

        for remove_idx in range(cur_rank):
            if time.time() - start_time > deadline - 10:
                break
            
            U_sub = np.delete(U, remove_idx, axis=1)
            V_sub = np.delete(V, remove_idx, axis=1)
            W_sub = np.delete(W, remove_idx, axis=1)
            
            T_decomp = np.einsum('ir,jr,kr->ijk', U_sub, V_sub, W_sub)
            T_error = T_true - T_decomp
            
            # Find a non-zero pivot
            nz_indices = np.nonzero(T_error)
            if len(nz_indices[0]) == 0:
                # Error is zero, which means we removed a redundant column? 
                # Or the decomposition was already rank target_rank? 
                # If T_error is 0, then U_sub, V_sub, W_sub is a valid decomposition of rank cur_rank-1.
                # But we want rank target_rank. If cur_rank-1 == target_rank, we are good.
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

            I0, J0, K0 = nz_indices[0][0], nz_indices[1][0], nz_indices[2][0]
            val = T_error[I0, J0, K0]
            
            a = T_error[:, J0, K0] / val
            b = T_error[I0, :, K0] / val
            c = T_error[I0, J0, :] / val
            
            T_reconstructed = np.einsum('i,j,k->ijk', a, b, c)
            
            if np.allclose(T_error, T_reconstructed, atol=1e-9):
                new_U = np.zeros((nm, target_rank))
                new_V = np.zeros((mp, target_rank))
                new_W = np.zeros((pn, target_rank))
                
                # Copy existing columns (cur_rank - 1 of them)
                # We need to fit them into target_rank.
                # If target_rank == cur_rank - 1, we just use U_sub.
                # If target_rank < cur_rank - 1, we can't just drop one column and add one, we need to drop more.
                # The function signature implies reducing by 1 step usually.
                # The caller asks for 16 from 17, so target_rank = 16, cur_rank = 17.
                # U_sub has 16 columns.
                new_U[:, :cur_rank-1] = U_sub
                new_V[:, :cur_rank-1] = V_sub
                new_W[:, :cur_rank-1] = W_sub
                
                # Add the new rank-1 component
                new_U[:, cur_rank-1] = a
                new_V[:, cur_rank-1] = b
                new_W[:, cur_rank-1] = c
                
                if verify(new_U, new_V, new_W):
                    return new_U, new_V, new_W
        
        return None
