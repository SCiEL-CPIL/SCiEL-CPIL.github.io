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
        nm, mp, pn = n*m, m*p, p*n
        
        # Precompute interaction matrices
        # M_a for U update: V[:, None, :] * W[None, :, :] -> (mp, pn, r) -> (mp*pn, r)
        M_U = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
        # M_b for V update: U[:, None, :] * W[None, :, :] -> (nm, pn, r) -> (nm*pn, r)
        M_V = (U[:, None, :] * W[None, :, :]).reshape(nm * pn, r)
        # M_c for W update: U[:, None, :] * V[None, :, :] -> (nm, mp, r) -> (nm*mp, r)
        M_W = (U[:, None, :] * V[None, :, :]).reshape(nm * mp, r)
        
        # Update U
        for a in range(nm):
            target = T[a, :, :].flatten()
            try:
                U[a, :], _, _, _ = np.linalg.lstsq(M_U.T, target, rcond=None)
            except:
                pass
        
        # Update V
        for b in range(mp):
            target = T[:, b, :].reshape(nm, pn).flatten()
            try:
                V[b, :], _, _, _ = np.linalg.lstsq(M_V.T, target, rcond=None)
            except:
                pass
        
        # Update W
        for c in range(pn):
            target = T[:, :, c].reshape(nm, mp).flatten()
            try:
                W[c, :], _, _, _ = np.linalg.lstsq(M_W.T, target, rcond=None)
            except:
                pass
        
        return U, V, W

    def polish(U, V, W, T, r):
        """Try to round to half-integers and check if exact"""
        # Try rounding to nearest integer
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        err_int = compute_error(U_int, V_int, W_int, T)
        if err_int < 1e-10:
            return U_int, V_int, W_int
        
        # Try rounding to nearest half-integer
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        err_half = compute_error(U_half, V_half, W_half, T)
        if err_half < 1e-10:
            return U_half, V_half, W_half
            
        return U, V, W

    # Search from lower ranks
    # Lower bound for 2x3x9 is likely around 27 (9 * rank(2x3x1))
    # rank(2x3x1) is 3. So 27 is a strong candidate.
    for r in range(27, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
            
        # Try multiple random initializations
        # Increase trials for lower ranks as they are harder to find
        trials = 50 if r < 35 else 30
        for trial in range(trials):
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
            
            for iteration in range(500):
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
                # Polish to get exact integer/half-integer if possible
                U, V, W = polish(U, V, W, T, r)
                # Final check
                if compute_error(U, V, W, T) < 1e-10:
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
