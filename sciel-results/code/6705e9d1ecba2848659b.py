import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the full tensor T as a 3D array
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Theoretical lower bound
    lb = max(nm, mp, pn)
    
    # For ALS: when updating U, for each row a, solve:
    # sum_t U[a,t] * V[b,t] * W[c,t] = T[a,b,c] for all b,c
    # This is a linear system: M_a @ U[a,:] = t_a
    # where M_a[b,c] = V[b,:] * W[c,:] (reshaped), t_a = T[a,:,:]
    
    def verify_decomposition(U_arr, V_arr, W_arr, tol=1e-10):
        """Verify that sum_t U[:,t] outer V[:,t] outer W[:,t] equals T"""
        # Compute reconstructed tensor
        recon = np.zeros((nm, mp, pn), dtype=float)
        for t in range(U_arr.shape[1]):
            recon += np.einsum('a,b,c->abc', U_arr[:, t], V_arr[:, t], W_arr[:, t])
        # Check max error
        max_err = np.max(np.abs(recon - T))
        return max_err < tol, max_err
    
    def try_als(r, n_trials=50, max_iter=200):
        """Try to find rank-r decomposition via ALS. Returns (U,V,W) or None."""
        np.random.seed(12345 + r * 1000)
        
        for trial in range(n_trials):
            if time.time() > deadline - 15:
                return None
            
            # Random initialization
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Precompute: for each pair (b,c), the vector V[b,:] * W[c,:]
            # Shape: (mp, pn, r)
            VW = np.einsum('bt,ct->bct', V, W)
            
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    return None
                
                # Update U: for each a, solve VW_reshaped @ U[a,:] = T[a,:,:].flatten()
                # VW_reshaped has shape (mp*pn, r), T[a,:,:].flatten() has shape (mp*pn,)
                VW_flat = VW.reshape(mp * pn, r)
                T_flat = T.reshape(nm, mp * pn)
                
                # Solve least squares for each row a
                # U[a,:] = argmin ||VW_flat @ U[a,:] - T_flat[a,:]||
                U_new = np.zeros((nm, r))
                for a in range(nm):
                    # Solve linear system
                    U_new[a, :] = np.linalg.lstsq(VW_flat, T_flat[a, :], rcond=None)[0]
                U = U_new
                
                # Update V: for each b, solve UW_reshaped @ V[b,:] = T_reshaped[b,:,:].flatten()
                # UW[a,c,t] = U[a,t] * W[c,t], shape (nm, pn, r)
                UW = np.einsum('at,ct->act', U, W)
                UW_flat = UW.reshape(nm * pn, r)
                # T reshaped: T[a,b,c], for fixed b, we need T[:,b,:].flatten()
                # T_reshaped[b] = T[:, b, :].flatten(), shape (nm*pn,)
                T_for_V = np.zeros((mp, nm * pn))
                for b in range(mp):
                    T_for_V[b, :] = T[:, b, :].flatten()
                
                V_new = np.zeros((mp, r))
                for b in range(mp):
                    V_new[b, :] = np.linalg.lstsq(UW_flat, T_for_V[b, :], rcond=None)[0]
                V = V_new
                
                # Update W: for each c, solve UV_reshaped @ W[c,:] = T_reshaped[c,:,:].flatten()
                # UV[a,b,t] = U[a,t] * V[b,t], shape (nm, mp, r)
                UV = np.einsum('at,bt->abt', U, V)
                UV_flat = UV.reshape(nm * mp, r)
                # T_reshaped[c] = T[:,:,c].flatten(), shape (nm*mp,)
                T_for_W = np.zeros((pn, nm * mp))
                for c in range(pn):
                    T_for_W[c, :] = T[:, :, c].flatten()
                
                W_new = np.zeros((pn, r))
                for c in range(pn):
                    W_new[c, :] = np.linalg.lstsq(UV_flat, T_for_W[c, :], rcond=None)[0]
                W = W_new
                
                # Check convergence
                if it % 10 == 9:
                    ok, err = verify_decomposition(U, V, W)
                    if ok:
                        return U, V, W
            
            # Final check after all iterations
            ok, err = verify_decomposition(U, V, W)
            if ok:
                return U, V, W
        
        return None
    
    # Try ranks from lower bound up to schoolbook rank - 1
    for r in range(lb, best_rank):
        if time.time() > deadline - 20:
            break
        
        result = try_als(r)
        if result is not None:
            U_arr, V_arr, W_arr = result
            # Convert to lists with rounded values for exactness
            U_list = [[round(val, 12) for val in row] for row in U_arr]
            V_list = [[round(val, 12) for val in row] for row in V_arr]
            W_list = [[round(val, 12) for val in row] for row in W_arr]
            
            # Final verification
            ok, err = verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list))
            if ok:
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                # Found a valid decomposition, continue to see if we can do better
                # But break early if we're running low on time
                if time.time() > deadline - 30:
                    break
            else:
                # Rounding broke it, try without rounding
                U_list = U_arr.tolist()
                V_list = V_arr.tolist()
                W_list = W_arr.tolist()
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                break
    
    return best
# EVOLVE-BLOCK-END
