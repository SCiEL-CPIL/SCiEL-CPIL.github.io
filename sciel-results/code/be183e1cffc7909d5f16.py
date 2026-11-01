import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 3, 4
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a low-rank decomposition using a structured approach
    # First, try the schoolbook as baseline (rank = n*m*p = 36)
    # Then try to reduce rank using known results for 3x3 * 3x4
    
    # The minimal rank for 3x3 * 3x4 matrix multiplication is at most 36 (schoolbook)
    # We can potentially do better with block methods
    
    # Let's try a direct ALS approach for various ranks
    def try_rank_als(r, max_iters=200, tol=1e-6, seed=42):
        """Try to find a rank-r decomposition using ALS."""
        if time.time() > deadline - 3:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Update U: for each row a, solve linear system
            for a in range(n*m):
                # System: sum_t U[a,t] * V[b,t] * W[c,t] = T[a,b,c] for all (b,c)
                # This is equivalent to: U[a,:] @ M_a = T[a,:,:].ravel()
                # where M_a[(b,c),t] = V[b,t] * W[c,t]
                
                # Build M_a efficiently
                # V has shape (m*p, r), W has shape (p*n, r)
                # We need outer product of each row of V with each row of W, flattened
                # M_a has shape (m*p*p*n, r)
                
                # Efficient construction:
                # For each t, M_a[:,t] = (V * W[None,:,:]).reshape(m*p*p*n)
                # where V is reshaped to (m*p, 1, r) and W to (1, p*n, r)
                
                # Actually, let's just build it directly since dimensions are small
                mp = m * p  # 12
                pn = p * n  # 12
                A_mat = np.zeros((mp * pn, r))
                idx = 0
                for b in range(mp):
                    for c in range(pn):
                        A_mat[idx, :] = V[b, :] * W[c, :]
                        idx += 1
                
                rhs = T[a, :, :].ravel()
                U[a, :], residuals, rank_a, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
            
            # Update V: for each row b, solve linear system
            for b in range(m*p):
                # System: sum_t V[b,t] * U[a,t] * W[c,t] = T[a,b,c] for all (a,c)
                mp2 = n * m  # 9
                pn2 = p * n  # 12
                A_mat = np.zeros((mp2 * pn2, r))
                idx = 0
                for a in range(n*m):
                    for c in range(p*n):
                        A_mat[idx, :] = U[a, :] * W[c, :]
                        idx += 1
                
                rhs = T[:, b, :].reshape(-1)
                V[b, :], residuals, rank_b, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
            
            # Update W: for each row c, solve linear system
            for c in range(p*n):
                mp3 = n * m  # 9
                mp4 = m * p  # 12
                A_mat = np.zeros((mp3 * mp4, r))
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        A_mat[idx, :] = U[a, :] * V[b, :]
                        idx += 1
                
                rhs = T[:, :, c].reshape(-1)
                W[c, :], residuals, rank_c, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
            
            # Check convergence
            recon = np.einsum('at,bt,ct->abc', U, V, W)
            err = np.abs(recon - T).max()
            if err < tol:
                break
        
        # Final check
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        err = np.abs(recon - T).max()
        if err > 1e-4:
            return None
        
        # Try to round to integer/half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        recon_r = np.einsum('at,bt,ct->abc', U_r, V_r, W_r)
        if np.abs(recon_r - T).max() < 1e-6:
            return (U_r, V_r, W_r)
        
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        recon_r = np.einsum('at,bt,ct->abc', U_r, V_r, W_r)
        if np.abs(recon_r - T).max() < 1e-6:
            return (U_r, V_r, W_r)
        
        return None
    
    # Try ranks from low to high
    best_result = None
    # Known: schoolbook is 36, try to find something better
    for r in range(25, 37):
        if time.time() > deadline - 5:
            break
        result = try_rank_als(r)
        if result is not None:
            best_result = (r, result)
            break
    
    if best_result is None:
        # Fallback to schoolbook
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = (r, (U, V, W))
    
    r, (U, V, W) = best_result
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
