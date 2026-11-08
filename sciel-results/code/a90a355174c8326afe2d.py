import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try Rank 9 for 2x3x3
    n_s, m_s, p_s = 2, 3, 3
    T = np.zeros((n_s * m_s, m_s * p_s, p_s * n_s))
    for i in range(n_s):
        for j in range(m_s):
            for k in range(p_s):
                T[i * m_s + j, j * p_s + k, k * n_s + i] = 1.0

    rank_s = 9
    U_s = np.zeros((n_s * m_s, rank_s))
    V_s = np.zeros((m_s * p_s, rank_s))
    W_s = np.zeros((p_s * n_s, rank_s))
    
    success = False
    for restart in range(50):
        if time.time() > deadline:
            break
            
        U_s[:] = np.random.randn(n_s * m_s, rank_s)
        V_s[:] = np.random.randn(m_s * p_s, rank_s)
        W_s[:] = np.random.randn(p_s * n_s, rank_s)
        
        # Normalize
        for r in range(rank_s):
            nU = np.linalg.norm(U_s[:, r])
            if nU > 1e-6:
                U_s[:, r] /= nU
                V_s[:, r] *= nU
            nV = np.linalg.norm(V_s[:, r])
            if nV > 1e-6:
                V_s[:, r] /= nV
                W_s[:, r] *= nV

        for it in range(2000):
            if time.time() > deadline:
                break
                
            # Update U
            for r in range(rank_s):
                # T = sum U V W
                # Fix V, W.
                # U[:, r] = (T - sum_{r'!=r} U_r' V_r' W_r') / (V_r * W_r)
                # This is element-wise division, which is only valid if V_r * W_r is non-zero.
                # Better: Least Squares.
                # T_flat (n_m * m_p * p_n)
                # U_r is vector of size n_m.
                # The term U_r \otimes V_r \otimes W_r.
                # We want to minimize || T - sum U_r V_r W_r ||^2.
                # This is a linear problem for U_r if V, W are fixed.
                
                # Residual
                Res = T.copy()
                for r2 in range(rank_s):
                    if r2 == r: continue
                    Res -= np.einsum('i,j,k->ijk', U_s[:, r2], V_s[:, r2], W_s[:, r2])
                
                # We want U_r such that U_r \otimes V_r \otimes W_r approx Res.
                # This is not a standard LS because of the tensor product.
                # However, if we fix V_r and W_r, the term is linear in U_r.
                # Res[i, j, k] = U_r[i] * V_r[j] * W_r[k]
                # U_r[i] = Res[i, j, k] / (V_r[j] * W_r[k])
                # This is only consistent if Res is a rank-1 tensor.
                # So we take the average over j,k where V_r[j]W_r[k] != 0.
                
                denom = np.outer(V_s[:, r], W_s[:, r])
                # Res is 3D. Flatten?
                # U_r[i] = sum_{j,k} Res[i,j,k] * V_r[j] * W_r[k] / sum_{j,k} V_r[j]^2 * W_r[k]^2
                num = np.einsum('ijk,j,k->i', Res, V_s[:, r], W_s[:, r])
                den = np.sum(V_s[:, r]**2) * np.sum(W_s[:, r]**2)
                if den > 1e-6:
                    U_s[:, r] = num / den
                    
            # Update V
            for r in range(rank_s):
                Res = T.copy()
                for r2 in range(rank_s):
                    if r2 == r: continue
                    Res -= np.einsum('i,j,k->ijk', U_s[:, r2], V_s[:, r2], W_s[:, r2])
                num = np.einsum('ijk,i,k->j', Res, U_s[:, r], W_s[:, r])
                den = np.sum(U_s[:, r]**2) * np.sum(W_s[:, r]**2)
                if den > 1e-6:
                    V_s[:, r] = num / den
                    
            # Update W
            for r in range(rank_s):
                Res = T.copy()
                for r2 in range(rank_s):
                    if r2 == r: continue
                    Res -= np.einsum('i,j,k->ijk', U_s[:, r2], V_s[:, r2], W_s[:, r2])
                num = np.einsum('ijk,i,j->k', Res, U_s[:, r], V_s[:, r])
                den = np.sum(U_s[:, r]**2) * np.sum(V_s[:, r]**2)
                if den > 1e-6:
                    W_s[:, r] = num / den
            
            # Check convergence
            Err = T - np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
            if np.linalg.norm(Err) < 1e-8:
                success = True
                break
                
        if success:
            break
            
    if success:
        # Round to integers/half-integers
        U_r = np.round(U_s * 2) / 2
        V_r = np.round(V_s * 2) / 2
        W_r = np.round(W_s * 2) / 2
        
        # Verify
        Err = T - np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
        if np.linalg.norm(Err) < 1e-6:
            # Build full 2x3x9
            U_full = np.zeros((n * m, 3 * rank_s))
            V_full = np.zeros((m * p, 3 * rank_s))
            W_full = np.zeros((p * n, 3 * rank_s))
            
            for block in range(3):
                # k indices: block*3 to block*3+2
                k_start = block * 3
                k_end = k_start + 3
                
                # U is shared, so just copy
                U_full[:, block * rank_s : (block+1) * rank_s] = U_r
                
                # V: rows j*p + k. j in 0..2, k in k_start..k_end
                # V_r has rows j*3 + k_local (0..2)
                for j in range(m_s):
                    for k_local in range(p_s):
                        v_idx_full = j * p + (k_start + k_local)
                        v_idx_sub = j * p_s + k_local
                        V_full[v_idx_full, block * rank_s : (block+1) * rank_s] = V_r[v_idx_sub]
                        
                # W: rows k*n + i. k in k_start..k_end, i in 0..1
                for k_local in range(p_s):
                    for i in range(n_s):
                        w_idx_full = (k_start + k_local) * n + i
                        w_idx_sub = k_local * n_s + i
                        W_full[w_idx_full, block * rank_s : (block+1) * rank_s] = W_r[w_idx_sub]
                        
            return {"U": U_full, "V": V_full, "W": W_full}
            
    return schoolbook()
