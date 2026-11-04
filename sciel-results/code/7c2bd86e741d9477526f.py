def _try_rank_reduction(U, V, W, n, m, p, target_rank):
    """Try to reduce the rank by eliminating terms."""
    R = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(n, m, R)
    V_arr = np.array(V, dtype=float).reshape(m, p, R)
    W_arr = np.array(W, dtype=float).reshape(p, n, R)
    
    # Compute T
    T = np.einsum('ijr,jkr,kir->ijk', U_arr, V_arr, W_arr, optimize=True)
    
    for r in range(R):
        if R - 1 < target_rank:
            break
        # Remove term r
        U_r = U_arr[:, :, r]
        V_r = V_arr[:, :, r]
        W_r = W_arr[:, :, r]
        
        T_res = T - np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r, optimize=True)
        
        # Solve for W' using remaining terms
        # For each (i,j,k): sum_{s≠r} U[i,j,s]*V[j,k,s]*W'[k,i,s] = T_res[i,j,k]
        
        # Build the linear system
        # Unknowns: W'[k,i,s] for s in {0,...,R-1}\{r}
        # Let's index them as (k, i, s') where s' is the index in the remaining set
        
        remaining = [s for s in range(R) if s != r]
        R_rem = len(remaining)
        
        # For each equation (i,j,k): sum_{s' } U[i,j,remaining[s']]*V[j,k,remaining[s']]*W'[k,i,s'] = T_res[i,j,k]
        
        # Build A matrix: 48 equations × 12*R_rem unknowns
        # Unknown index: (k, i, s') -> k*n + i*R_rem + s'... 
        # Actually, let's use a flat index: for each (k,i,s'), the index is ((k*n + i)*R_rem + s')
        
        n_eq = n * m * p  # 48
        n_unk = p * n * R_rem  # 12 * R_rem
        
        A = np.zeros((n_eq, n_unk))
        b = np.zeros(n_eq)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    b[eq_idx] = T_res[i, j, k]
                    for si, s in enumerate(remaining):
                        unk_idx = (k * n + i) * R_rem + si
                        A[eq_idx, unk_idx] = U_arr[i, j, s] * V_arr[j, k, s]
        
        # Solve
        W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        
        # Check residual
        if len(residuals) > 0 and residuals[0] < 1e-8:
            # Found a valid rank-(R-1) decomposition
            W_new = W_flat.reshape(p, n, R_rem)
            U_new = np.delete(U_arr, r, axis=2)
            V_new = np.delete(V_arr, r, axis=2)
            
            # Verify
            T_check = np.einsum('ijr,jkr,kir->ijk', U_new, V_new, W_new, optimize=True)
            if np.allclose(T_check, 1.0, atol=1e-8):
                return U_new, V_new, W_new
    
    return None
