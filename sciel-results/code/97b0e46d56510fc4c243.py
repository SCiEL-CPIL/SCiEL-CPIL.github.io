import numpy as np
import time

def construct():
    # Dimensions
    n, m, p = 2, 3, 9
    
    # Target Tensor T[a,b,c] for M_{2,3,9}
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook fallback
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

    deadline = time.time() + 140  # Safety margin

    # Strategy: Find rank 11 for M_{2,3,3}, then expand to rank 33 for M_{2,3,9}
    n_s, m_s, p_s = 2, 3, 3
    T_s = np.zeros((n_s*m_s, m_s*p_s, p_s*n_s))
    for i in range(n_s):
        for j in range(m_s):
            for k in range(p_s):
                T_s[i*m_s+j, j*p_s+k, k*n_s+i] = 1.0

    def als_step_small(U, V, W, T, r):
        nm, mp, pn = n_s*m_s, m_s*p_s, p_s*n_s
        # Update U
        # M_a is (mp*pn, r), target is (mp*pn,)
        # We can precompute V*W once
        VW = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
        for a in range(nm):
            target = T[a, :, :].flatten()
            try:
                U[a, :], _, _, _ = np.linalg.lstsq(VW.T, target, rcond=None)
            except:
                pass
        
        # Update V
        UW = (U[:, None, :] * W[None, :, :]).reshape(nm * pn, r)
        for b in range(mp):
            target = T[:, b, :].reshape(nm, pn).flatten()
            try:
                V[b, :], _, _, _ = np.linalg.lstsq(UW.T, target, rcond=None)
            except:
                pass
                
        # Update W
        UV = (U[:, None, :] * V[None, :, :]).reshape(nm * mp, r)
        for c in range(pn):
            target = T[:, :, c].reshape(nm, mp).flatten()
            try:
                W[c, :], _, _, _ = np.linalg.lstsq(UV.T, target, rcond=None)
            except:
                pass
        return U, V, W

    def compute_error_small(U, V, W, T):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_approx - T)**2)

    # Find minimal rank for M_{2,3,3}
    best_rank_s = n_s * m_s * p_s
    best_U_s, best_V_s, best_W_s = None, None, None
    
    # Try rank 11 for M_{2,3,3}
    # Known minimal rank is 11
    r_target = 11
    success = False
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline - 30:
            break
        np.random.seed(42 + trial)
        U = np.random.randn(n_s*m_s, r_target)
        V = np.random.randn(m_s*p_s, r_target)
        W = np.random.randn(p_s*n_s, r_target)
        
        # Normalize
        for c in range(r_target):
            norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
            if norm > 1e-10:
                U[:, c] /= norm
                V[:, c] /= norm
                W[:, c] /= norm
                
        prev_error = compute_error_small(U, V, W, T_s)
        
        max_iters = 100
        for it in range(max_iters):
            if time.time() > deadline - 30:
                break
            U, V, W = als_step_small(U, V, W, T_s, r_target)
            error = compute_error_small(U, V, W, T_s)
            if error < 1e-10:
                success = True
                break
            if error > 1e6:
                break
            if abs(error - prev_error) < 1e-14:
                if error < 1e-8:
                    success = True
                break
            prev_error = error
            
        if success:
            best_rank_s = r_target
            best_U_s, best_V_s, best_W_s = U, V, W
            break

    # Construct M_{2,3,9} from M_{2,3,3}
    if best_U_s is not None:
        rank_candidate = 3 * best_rank_s
        if rank_candidate < best_rank:
            U_new = np.hstack([best_U_s, best_U_s, best_U_s])
            
            V_new = np.zeros((m*p, 3*best_rank_s))
            for block in range(3):
                for j in range(m_s):
                    for k in range(p_s):
                        row_new = j * p + (block * p_s + k)
                        row_s = j * p_s + k
                        V_new[row_new, block*best_rank_s : (block+1)*best_rank_s] = best_V_s[row_s, :]
                        
            W_new = np.zeros((p*n, 3*best_rank_s))
            for block in range(3):
                for k in range(p_s):
                    for i in range(n_s):
                        row_new = (block * p_s + k) * n + i
                        row_s = k * n_s + i
                        W_new[row_new, block*best_rank_s : (block+1)*best_rank_s] = best_W_s[row_s, :]
                        
            # Verify
            T_approx = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
            error = np.sum((T_approx - T)**2)
            if error < 1e-6:
                best_rank = rank_candidate
                best_U, best_V, best_W = U_new, V_new, W_new

    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
