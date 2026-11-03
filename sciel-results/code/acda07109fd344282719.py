import numpy as np
import time

def _als_update_U(U, V, W, T_target, n, m, p, lam=0.01):
    """
    Update U given V, W, and target tensor T.
    Optimized using pre-computed outer products.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    
    # M[b*pn+c, k] = V[b, k] * W[c, k]
    M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
    
    MtM = M.T @ M + lam * np.eye(r)
    
    # T_target_flat: (nm, mp*pn)
    T_flat = T_target.reshape(nm, mp * pn)
    
    Y = M.T @ T_flat.T
    # Solve MtM @ U_new.T = Y
    U_new_T = np.linalg.solve(MtM, Y)
    U_new = U_new_T.T
    
    return U_new

def _als_update_V(U, V, W, T_target, n, m, p, lam=0.01):
    """
    Update V given U, W, and target tensor T.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = V.shape[1]
    
    # M[a*pn+c, k] = U[a, k] * W[c, k]
    M = (U[:, None, :] * W[None, :, :]).reshape(nm * pn, r)
    
    MtM = M.T @ M + lam * np.eye(r)
    
    # T_target.transpose(1, 0, 2) -> (mp, nm, pn)
    T_flat_T = T_target.transpose(1, 0, 2).reshape(mp, nm * pn)
    
    Y = M.T @ T_flat_T.T
    V_new_T = np.linalg.solve(MtM, Y)
    V_new = V_new_T.T
    
    return V_new

def _als_update_W(U, V, W, T_target, n, m, p, lam=0.01):
    """
    Update W given U, V, and target tensor T.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = W.shape[1]
    
    # M[a*mp+b, k] = U[a, k] * V[b, k]
    M = (U[:, None, :] * V[None, :, :]).reshape(nm * mp, r)
    
    MtM = M.T @ M + lam * np.eye(r)
    
    # T_target.transpose(2, 0, 1) -> (pn, nm, mp)
    T_flat_T = T_target.transpose(2, 0, 1).reshape(pn, nm * mp)
    
    Y = M.T @ T_flat_T.T
    W_new_T = np.linalg.solve(MtM, Y)
    W_new = W_new_T.T
    
    return W_new

def construct():
    n, m, p = 3, 5, 6
    nm, mp, pn = n * m, m * p, p * n
    T_target = np.zeros((nm, mp, pn))
    
    # Populate T_target
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i', j==j', k==k'
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    # Trivial decomposition rank 90
    r_trivial = n * m * p
    U_trivial = np.zeros((nm, r_trivial))
    V_trivial = np.zeros((mp, r_trivial))
    W_trivial = np.zeros((pn, r_trivial))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r_idx = i * m * p + j * p + k
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, r_idx] = 1.0
                V_trivial[b, r_idx] = 1.0
                W_trivial[c, r_idx] = 1.0
                
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    best_rank = r_trivial
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Try to find lower rank decompositions
    # We search downwards from 90
    for r in range(89, 0, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + r * 100 + restart)
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Run ALS
            for it in range(100):
                if time.time() > deadline:
                    break
                U = _als_update_U(U, V, W, T_target, n, m, p)
                V = _als_update_V(U, V, W, T_target, n, m, p)
                W = _als_update_W(U, V, W, T_target, n, m, p)
                
            # Calculate error
            # Reconstruct T_approx
            # T_approx[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
            # Efficient check:
            # M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
            # T_approx_flat = U @ M.T -> (nm, mp*pn)
            M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
            T_approx_flat = U @ M.T
            T_target_flat = T_target.reshape(nm, mp * pn)
            error = np.linalg.norm(T_approx_flat - T_target_flat)
            
            if error < 1e-6:
                # Found an exact decomposition
                # Attempt to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Check if rounded version is also exact
                M_rounded = (V_rounded[:, None, :] * W_rounded[None, :, :]).reshape(mp * pn, r)
                T_approx_rounded_flat = U_rounded @ M_rounded.T
                error_rounded = np.linalg.norm(T_approx_rounded_flat - T_target_flat)
                
                if error_rounded < 1e-6:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                else:
                    best_U, best_V, best_W = U, V, W
                    
                best_rank = r
                found = True
                break # Break restart loop, try lower rank
        
        if found:
            continue
        else:
            # If we failed for this rank, we might still have a valid lower rank later?
            # But if rank r fails, rank r-1 might succeed?
            # Usually, if rank r is not exact, r-1 is less likely to be exact unless the structure allows it.
            # However, we should continue searching down to be safe, or stop?
            # Given time constraints, we might just break if we haven't found anything for a while?
            # No, just continue.
            pass

    return {
        'U': best_U,
        'V': best_V,
        'W': best_W
    }
