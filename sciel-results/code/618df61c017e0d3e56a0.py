import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    
    # Build the target tensor
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    deadline = time.time() + 135  # 135 second budget (150s limit - 15s safety)
    
    def compute_error(U, V, W, T):
        """Compute squared error efficiently using einsum"""
        # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_approx - T)**2)
    
    def als_step(U, V, W, T, r):
        """One ALS iteration: update U, V, W in sequence"""
        n, m, p = 2, 3, 9
        nm, mp, pn = n*m, m*p, p*n
        
        # Update U: for each a, solve M_a^T @ U[a,:] = T[a,:,:].flatten()
        # where M_a[b*pn+c, r] = V[b,r] * W[c,r]
        for a in range(nm):
            # Build M_a efficiently
            # M_a is (mp*pn) x r, M_a[b*pn+c, r] = V[b,r] * W[c,r]
            # This can be written as: reshape V to (mp, r), W to (pn, r)
            # M_a[b*pn+c, r] = V[b,r] * W[c,r]
            # Using broadcasting: V[:, None, :] * W[None, :, :] gives (mp, pn, r)
            # Reshape to (mp*pn, r)
            M_a = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
            target = T[a, :, :].flatten()
            try:
                U[a, :], _, _, _ = np.linalg.lstsq(M_a.T, target, rcond=None)
            except:
                pass
        
        # Update V: for each b, solve M_b^T @ V[b,:] = T[:,b,:].flatten()
        # where M_b[a*pn+c, r] = U[a,r] * W[c,r]
        for b in range(mp):
            M_b = (U[:, None, :] * W[None, :, :]).reshape(nm * pn, r)
            target = T[:, b, :].reshape(nm, pn).flatten()
            try:
                V[b, :], _, _, _ = np.linalg.lstsq(M_b.T, target, rcond=None)
            except:
                pass
        
        # Update W: for each c, solve M_c^T @ W[c,:] = T[:,:,c].flatten()
        # where M_c[a*mp+b, r] = U[a,r] * V[b,r]
        for c in range(pn):
            M_c = (U[:, None, :] * V[None, :, :]).reshape(nm * mp, r)
            target = T[:, :, c].reshape(nm, mp).flatten()
            try:
                W[c, :], _, _, _ = np.linalg.lstsq(M_c.T, target, rcond=None)
            except:
                pass
        
        return U, V, W
    
    for r in range(33, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
            
        # Try multiple random initializations
        for trial in range(30):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(42 + r * 1000 + trial)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Normalize
            for col_idx in range(r):
                norm = np.sqrt(np.sum(U[:, col_idx]**2) + np.sum(V[:, col_idx]**2) + np.sum(W[:, col_idx]**2))
                if norm > 1e-10:
                    U[:, col_idx] /= norm
                    V[:, col_idx] /= norm
                    W[:, col_idx] /= norm
            
            success = False
            prev_error = compute_error(U, V, W, T)
            
            for iteration in range(300):
                if time.time() > deadline - 10:
                    break
                    
                U, V, W = als_step(U, V, W, T, r)
                
                # Check convergence
                error = compute_error(U, V, W, T)
                
                if error < 1e-10:
                    success = True
                    break
                elif error > 1e6:
                    break  # Diverged
                elif abs(error - prev_error) < 1e-15:
                    # Stagnated
                    if error < 1e-6:
                        # Close enough, try to polish
                        success = True
                    break
                prev_error = error
            
            if success:
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break  # Found at this rank, no need to try higher r
        
        if best_rank == r:
            break  # Found at this rank
    
    # Convert to lists
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
