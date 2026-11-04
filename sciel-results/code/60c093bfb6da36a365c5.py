import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 20s safety margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k', k''*n+i''] = 1 iff j==j' and i==i'' and k==k'
    Nm = n * m
    Mp = m * p
    Pn = p * n
    
    # T[a, b, c] where a = i*m+j, b = j'*p+k', c = k''*n+i''
    # T[a,b,c] = 1 iff j==j' and i==i'' and k==k'
    T = np.zeros((Nm, Mp, Pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    def schoolbook():
        rank = rank_schoolbook
        U = np.zeros((Nm, rank))
        V = np.zeros((Mp, rank))
        W = np.zeros((Pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Try alternating minimization for various ranks
    best_rank = rank_schoolbook
    best_U, best_V, best_W = schoolbook()
    
    # Try ranks close to schoolbook first, as lower ranks are unlikely for small matrices
    # and expensive to verify.
    ranks_to_try = list(range(rank_schoolbook - 1, max(rank_schoolbook - 10, 59), -1))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Only 1 trial per rank to save time
        np.random.seed(42 + target_rank)
        U = np.random.randn(Nm, target_rank)
        V = np.random.randn(Mp, target_rank)
        W = np.random.randn(Pn, target_rank)
        
        # Alternating minimization
        max_iters = 50
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
                # Fix V, W; solve for U
                # For each column c of U, we want to minimize ||T[:,b,c'] - U[:,c]*V[:,c]*W[:,c']||... 
                # Actually, the model is: T[a,b,c_idx] = sum_r U[a,r] * V[b,r] * W[c_idx,r]
                # This is not linear in U when we fix V and W, because W[c_idx,r] couples them.
                
                # Let me reformulate. For fixed V and W, define for each pair (b, c_idx):
                # Vb_Wc = V[b,:].elementwise(W[c_idx,:])  (elementwise product)
                # Then T[:, b, c_idx] = U @ Vb_Wc  (as a vector in a-space)
                
                # So for each (b, c_idx), we have: T[:, b, c_idx] = U @ (V[b,:] * W[c_idx,:])
                # This is a linear system in U for each (b, c_idx).
                
                # Combine all (b, c_idx) pairs into one big linear system.
                # Let M be a matrix of size (Mp*Pn, target_rank) where M[(b,c_idx), r] = V[b,r] * W[c_idx,r]
                # Then for each a: T[a, b, c_idx] = sum_r U[a,r] * M[(b,c_idx), r]
                # So T_flat = U @ M^T where T_flat has shape (Nm, Mp*Pn)
                
                # Actually: T[a, b, c_idx] = sum_r U[a, r] * V[b, r] * W[c_idx, r]
                # Let me define for each r: the (b, c_idx) slice is V[:,r] * W[:,r] (outer product reshaped)
                # So T[a, :, :] = sum_r U[a, r] * (V[:,r] outer W[:,r]).reshape(Mp, Pn)
                
                # For fixed V, W, this is linear in U.
                # Build the design matrix.
                
                if time.time() > deadline - 30:
                    break
                
                # Reshape T to (Nm, Mp*Pn)
                T_flat = T.reshape(Nm, Mp * Pn)
                
                # Correct: M[b*Pn+c, r] = V[b,r] * W[c,r]
                # Use broadcasting for speed
                # V is (Mp, R), W is (Pn, R)
                # We want M of shape (Mp, Pn, R) where M[b, c, r] = V[b, r] * W[c, r]
                M = V[:, np.newaxis, :] * W[np.newaxis, :, :]
                M_flat = M.reshape(Mp * Pn, target_rank)
                
                # Now T_flat[a, b*Pn+c] = sum_r U[a,r] * M[b*Pn+c, r]
                # So T_flat = U @ M.T
                # Solve for U: U = T_flat @ M^+ (pseudo-inverse)
                
                # T_flat (Nm, Mp*Pn) = U (Nm, R) @ M_flat.T (R, Mp*Pn)
                # Transpose: T_flat.T (Mp*Pn, Nm) = M_flat (Mp*Pn, R) @ U.T (R, Nm)
                # Solve M_flat X = T_flat.T for X (R, Nm)
                try:
                    U_T, residuals, rank_u, s = np.linalg.lstsq(M_flat, T_flat.T, rcond=None)
                    U = U_T.T
                except:
                    pass
                
                # Update V
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # Fix U, W.
                # T_flat (Nm, Mp*Pn)
                # For fixed U, W, define N[b, c, r] = U[a, r] W[c, r]? No, U depends on a.
                # T[a, b, c] = sum_r (U[a, r] W[c, r]) V[b, r]
                # Let K[a, c, r] = U[a, r] W[c, r].
                # T[a, b, c] = sum_r K[a, c, r] V[b, r]
                # This is not a simple linear system for V because K depends on a and c.
                # We can group by b.
                # For a fixed b, T[:, b, :] (Nm, Pn) = V[b, :] @ (U * W) ...
                # Let's reshape.
                # T is (Nm, Mp, Pn).
                # For each b, we have a matrix T_b of shape (Nm, Pn).
                # T_b[a, c] = sum_r U[a, r] V[b, r] W[c, r]
                # T_b = (U * V[b, :]) @ W.T ?
                # (U * V[b, :]) is (Nm, R) where element (a, r) is U[a, r] * V[b, r].
                # Let Y_b = U * V[b, :] (broadcasting).
                # T_b = Y_b @ W.T
                # We want to find V[b, :].
                # T_b = U @ diag(V[b, :]) @ W.T
                # This is not linear in V[b, :] in a simple matrix multiply way for lstsq unless we vectorize.
                # Vectorize T_b to vec(T_b) (Nm*Pn, 1).
                # vec(T_b) = (W.T ⊗ U) vec(V[b, :])?
                # Let's check: T_b[a, c] = sum_r U[a, r] V[b, r] W[c, r].
                # vec(T_b) = sum_r V[b, r] (U[:, r] ⊗ W[c, r]?) No.
                # vec(T_b) = sum_r V[b, r] (U[:, r] * W[:, r].T) ? No.
                # Let's build the design matrix for V.
                # D_V[b, (a, c), r] = U[a, r] * W[c, r].
                # Then T_flat_b = V[b, :] @ D_V[b, :, :].T ?
                # T_flat_b[(a, c)] = sum_r V[b, r] D_V[b, (a, c), r]
                # Yes.
                
                # This is getting complex. Given the time limit and likelihood of rank 120, 
                # simple alternating minimization might be overkill or prone to bugs.
                # However, we must fix the code to run.
                
                # Let's just compute error and break if no improvement.
                # If we don't update V and W, U will oscillate or converge to a local min that doesn't fit T.
                # Actually, if we only update U, V and W stay random, so U will just try to fit T with fixed random V, W.
                # This is equivalent to finding the best rank-R approximation of T using the fixed subspace spanned by V, W.
                # It won't find the true decomposition unless V, W are already close.
                
                # To make it work, we MUST update V and W.
                # Let's implement the full update.
                
                # Update V
                # T_flat (Nm, Mp, Pn)
                # For each b, solve for V[b, :].
                # T[:, b, :] (Nm, Pn) = (U * V[b, :]) @ W.T
                # Let A_b = U * W.T ? No.
                # T[:, b, :] = U @ diag(V[b, :]) @ W.T
                # Let's vectorize.
                # T_vec_b = (W.T ⊗ U) v_b ?
                # Let's build the matrix M_V of shape (Mp, Nm*Pn, R).
                # M_V[b, a*Pn+c, r] = U[a, r] * W[c, r].
                # Then T_flat (Nm, Mp*Pn) -> reshape to (Mp, Nm*Pn)? No, T is (Nm, Mp, Pn).
                # Let's permute T to (Mp, Nm, Pn).
                T_perm = np.transpose(T, (1, 0, 2)) # (Mp, Nm, Pn)
                T_perm_flat = T_perm.reshape(Mp, Nm * Pn)
                
                # Build M_V: (Mp, Nm*Pn, R)
                # M_V[b, a*Pn+c, r] = U[a, r] * W[c, r]
                # U is (Nm, R), W is (Pn, R)
                # U[:, np.newaxis, :] * W[np.newaxis, :, :] -> (Nm, Pn, R)
                # Reshape to (Nm*Pn, R)
                M_V_slice = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(Nm * Pn, target_rank)
                # Since U and W are fixed, M_V_slice is the same for all b?
                # Yes, M_V[b, :, :] = M_V_slice for all b.
                # So T_perm_flat (Mp, Nm*Pn) = V (Mp, R) @ M_V_slice.T (R, Nm*Pn)
                # Transpose: T_perm_flat.T (Nm*Pn, Mp) = M_V_slice (Nm*Pn, R) @ V.T (R, Mp)
                # Solve M_V_slice X = T_perm_flat.T for X (R, Mp)
                try:
                    V_T, _, _, _ = np.linalg.lstsq(M_V_slice, T_perm_flat.T, rcond=None)
                    V = V_T.T
                except:
                    pass

                # Update W
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # Permute T to (Pn, Nm, Mp).
                T_perm2 = np.transpose(T, (2, 0, 1)) # (Pn, Nm, Mp)
                T_perm2_flat = T_perm2.reshape(Pn, Nm * Mp)
                
                # M_W_slice: (Nm*Mp, R)
                # M_W_slice[a*Mp+b, r] = U[a, r] * V[b, r]
                M_W_slice = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(Nm * Mp, target_rank)
                # T_perm2_flat (Pn, Nm*Mp) = W (Pn, R) @ M_W_slice.T
                # Transpose: T_perm2_flat.T (Nm*Mp, Pn) = M_W_slice (Nm*Mp, R) @ W.T (R, Pn)
                try:
                    W_T, _, _, _ = np.linalg.lstsq(M_W_slice, T_perm2_flat.T, rcond=None)
                    W = W_T.T
                except:
                    pass

                # Compute error
                T_pred = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_pred - T) / np.linalg.norm(T)
                
                if err < 1e-6:
                    # Exact decomposition found
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
                
                if err > prev_err * 0.999:
                    break
                prev_err = err
        
        if best_rank <= target_rank:
            break
    
    # Snap to half-integers
    U_snap = np.round(best_U * 2) / 2
    V_snap = np.round(best_V * 2) / 2
    W_snap = np.round(best_W * 2) / 2
    
    # Verify
    T_pred = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
    err = np.max(np.abs(T_pred - T))
    
    if err > 1e-9:
        # Fallback to schoolbook
        U_snap, V_snap, W_snap = schoolbook()
        best_rank = rank_schoolbook
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U_snap]
    V_list = [[float(x) for x in row] for row in V_snap]
    W_list = [[float(x) for x in row] for row in W_snap]
    
    return {"U": U_list, "V": V_list, "W": W_list}
