import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = p * n
    
    # Schoolbook rank
    rank_school = nm * p # 3*3*6 = 54? No, nm*p is 54.
    # Actually schoolbook rank is n*m*p = 54.
    
    # Build the target MM Tensor T
    # T is a 3D array of shape (nm, mp, np_)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k) else 0
    T = np.zeros((nm, mp, np_), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def check_decomposition(U, V, W):
        # U: (nm, R), V: (mp, R), W: (np_, R)
        # Compute sum_r U[:, r] V[:, r] W[:, r]
        # This is not a simple matmul.
        # T_calc[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        R = U.shape[1]
        # We can use einsum
        # 'ar,br,cr -> abc'
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check exactness
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        
        # Check integer/half-integer
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        if not np.allclose(U, U_r, atol=1e-9): return False
        if not np.allclose(V, V_r, atol=1e-9): return False
        if not np.allclose(W, W_r, atol=1e-9): return False
        
        # Re-verify with rounded values
        T_calc_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if not np.allclose(T_calc_r, T, atol=1e-9):
            return False
            
        return True

    def als_decomposition(rank, max_iters=50, tol=1e-8):
        # Random init
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(np_, rank)
        
        prev_err = float('inf')
        for _ in range(max_iters):
            # Solve for U
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # For fixed V, W, this is linear in U.
            # T[:, b, c] = U @ (V[:, b, :] * W[:, c, :])^T ? No.
            # T[a, b, c] = sum_r U[a, r] (V[b, r] W[c, r])
            # Let M_U[b, c, r] = V[b, r] W[c, r]
            # T[a, b, c] = sum_r U[a, r] M_U[b, c, r]
            # Reshape T to (nm, mp*np_)
            # Reshape M_U to (mp*np_, rank)
            T_flat = T.reshape(nm, -1)
            M_U = (V[:, :, None] * W[None, :, :]).reshape(rank, -1).T # (mp*np_, rank)
            # Solve U @ M_U.T = T_flat
            U, *_ = np.linalg.lstsq(M_U, T_flat.T, rcond=None)
            
            # Solve for V
            # T[a, b, c] = sum_r V[b, r] (U[a, r] W[c, r])
            T_flat_V = T.transpose(1, 0, 2).reshape(mp, -1) # (mp, nm*np_)
            M_V = (U[:, :, None] * W[None, :, :]).reshape(rank, -1).T # (nm*np_, rank)
            V, *_ = np.linalg.lstsq(M_V, T_flat_V.T, rcond=None)
            
            # Solve for W
            # T[a, b, c] = sum_r W[c, r] (U[a, r] V[b, r])
            T_flat_W = T.transpose(2, 0, 1).reshape(np_, -1) # (np_, nm*mp)
            M_W = (U[:, :, None] * V[None, :, :]).reshape(rank, -1).T # (nm*mp, rank)
            W, *_ = np.linalg.lstsq(M_W, T_flat_W.T, rcond=None)
            
            # Calculate error
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_calc - T)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        if check_decomposition(U, V, W):
            return U, V, W
        return None

    best_result = None
    best_rank = rank_school
    
    # Fallback to schoolbook
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((np_, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    start_time = time.time()
    deadline = 100.0
    
    # Try ranks from 23 to 46
    for rank in range(23, 47):
        if time.time() > deadline:
            break
        
        # Multiple trials
        trials = 5
        for _ in range(trials):
            if time.time() > deadline:
                break
            res = als_decomposition(rank, max_iters=100)
            if res is not None:
                U, V, W = res
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                # Found a lower rank, break out?
                # We want the MINIMAL rank. So break.
                return best_result
                
    return best_result
