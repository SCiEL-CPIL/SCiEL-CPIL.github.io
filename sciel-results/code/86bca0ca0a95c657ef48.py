import numpy as np
import time

DIMENSIONS = [2, 4, 7]
DEADLINE = 135.0  # safety margin under 150s

def _time_left():
    return DEADLINE - time.time()

def _tensor_from(U, V, W):
    # U: (a, r), V: (b, r), W: (c, r) -> tensor (a,b,c)
    return np.einsum('ar,br,cr->abc', U, V, W)

def _check(U, V, W, T, tol=1e-6):
    R = _tensor_from(U, V, W)
    return np.allclose(R, T, atol=tol, rtol=tol)

def _round_half(M):
    return np.round(M * 2.0) / 2.0

def _get_strassen_222():
    r = 7
    U = np.zeros((4, r))
    V = np.zeros((4, r))
    W = np.zeros((4, r))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_school_221():
    r = 4
    U = np.zeros((4, r))
    V = np.zeros((2, r))
    W = np.zeros((2, r))
    
    for c in range(r):
        i = c // 2
        j = c % 2
        U[i * 2 + j, c] = 1.0
        V[j, c] = 1.0
        W[i, c] = 1.0
        
    return U, V, W

def _build_strassen_decomposition():
    n, m, p = 2, 4, 7
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    for j_off in [0, 2]:
        # p in [0, 2): Strassen 2x2x2
        U_sub, V_sub, W_sub = _get_strassen_222()
        r_sub = U_sub.shape[1]
        
        for i in range(2):
            for j in range(2):
                U[i * 4 + (j_off + j), col_offset:col_offset + r_sub] = U_sub[i * 2 + j, :]
        for j in range(2):
            for k in range(2):
                V[(j_off + j) * 7 + k, col_offset:col_offset + r_sub] = V_sub[j * 2 + k, :]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i, col_offset:col_offset + r_sub] = W_sub[k * 2 + i, :]
        col_offset += r_sub
        
        # p in [2, 4): Strassen 2x2x2
        U_sub, V_sub, W_sub = _get_strassen_222()
        r_sub = U_sub.shape[1]
        
        for i in range(2):
            for j in range(2):
                U[i * 4 + (j_off + j), col_offset:col_offset + r_sub] = U_sub[i * 2 + j, :]
        for j in range(2):
            for k in range(2):
                V[(j_off + j) * 7 + (k + 2), col_offset:col_offset + r_sub] = V_sub[j * 2 + k, :]
        for k in range(2):
            for i in range(2):
                W[(k + 2) * 2 + i, col_offset:col_offset + r_sub] = W_sub[k * 2 + i, :]
        col_offset += r_sub
        
        # p in [4, 6): Strassen 2x2x2
        U_sub, V_sub, W_sub = _get_strassen_222()
        r_sub = U_sub.shape[1]
        
        for i in range(2):
            for j in range(2):
                U[i * 4 + (j_off + j), col_offset:col_offset + r_sub] = U_sub[i * 2 + j, :]
        for j in range(2):
            for k in range(2):
                V[(j_off + j) * 7 + (k + 4), col_offset:col_offset + r_sub] = V_sub[j * 2 + k, :]
        for k in range(2):
            for i in range(2):
                W[(k + 4) * 2 + i, col_offset:col_offset + r_sub] = W_sub[k * 2 + i, :]
        col_offset += r_sub
        
        # p in [6, 7): Schoolbook 2x2x1
        U_sub, V_sub, W_sub = _get_school_221()
        r_sub = U_sub.shape[1]
        
        for i in range(2):
            for j in range(2):
                U[i * 4 + (j_off + j), col_offset:col_offset + r_sub] = U_sub[i * 2 + j, :]
        for j in range(2):
            V[(j_off + j) * 7 + 6, col_offset:col_offset + r_sub] = V_sub[j, :]
        for i in range(2):
            W[6 * 2 + i, col_offset:col_offset + r_sub] = W_sub[i, :]
        col_offset += r_sub
    
    return U, V, W

def _build_target_tensor():
    n, m, p = 2, 4, 7
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _alt_minimize_fast(U, V, W, T, n_trials=10, max_iter=50, r=None):
    """
    Fast alternating minimization exploiting separability of least squares.
    """
    a, b, c = T.shape
    if r is None:
        r = U.shape[1]
    
    for trial in range(n_trials):
        if _time_left() < 3:
            return None
        
        if trial == 0:
            U_curr = U.copy()
            V_curr = V.copy()
            W_curr = W.copy()
        else:
            rng = np.random.RandomState(trial * 137 + 42)
            U_curr = U + rng.randn(a, r) * 0.1
            V_curr = V + rng.randn(b, r) * 0.1
            W_curr = W + rng.randn(c, r) * 0.1
        
        prev_err = np.inf
        for it in range(max_iter):
            if _time_left() < 2:
                return None
            
            # Optimize W
            # For each c_val, solve A_c w_c = t_c
            # A_c[i*b + j, d] = U[i,d] * V[j,d]
            # t_c = T[:,:,c_val].flatten()
            W_new = W_curr.copy()
            for c_val in range(c):
                A_c = np.outer(U_curr, V_curr).reshape(a*b, r) # Wait, outer(U,V) is (a,b,r)? No.
                # U is (a,r), V is (b,r).
                # We need matrix of size (a*b, r).
                # Row index i*b+j, col d: U[i,d]*V[j,d]
                A_c = np.einsum('id,jd->ijd', U_curr, V_curr).reshape(a*b, r)
                t_c = T[:,:,c_val].reshape(a*b)
                try:
                    W_new[c_val, :] = np.linalg.lstsq(A_c, t_c, rcond=None)[0]
                except:
                    pass
            
            # Optimize U
            # For each i_val, solve B_i u_i = t_i
            # B_i[j*c + c_val, d] = V[j,d] * W[c_val,d]
            # t_i = T[i_val,:,:].flatten() (size b*c)
            U_new = U_curr.copy()
            for i_val in range(a):
                B_i = np.einsum('jd,kd->jkd', V_curr, W_new).reshape(b*c, r)
                t_i = T[i_val,:,:].reshape(b*c)
                try:
                    U_new[i_val, :] = np.linalg.lstsq(B_i, t_i, rcond=None)[0]
                except:
                    pass
            
            # Optimize V
            # For each j_val, solve C_j v_j = t_j
            # C_j[i*c + c_val, d] = U[i,d] * W[c_val,d]
            # t_j = T[:,j_val,:].flatten() (size a*c)
            V_new = V_curr.copy()
            for j_val in range(b):
                C_j = np.einsum('id,kd->ikd', U_new, W_new).reshape(a*c, r)
                t_j = T[:,j_val,:].reshape(a*c)
                try:
                    V_new[j_val, :] = np.linalg.lstsq(C_j, t_j, rcond=None)[0]
                except:
                    pass
            
            U_curr, V_curr, W_curr = U_new, V_new, W_new
            
            # Compute error
            T_rec = _tensor_from(U_curr, V_curr, W_curr)
            err = np.linalg.norm(T_rec - T)
            if err < 1e-10:
                break
            if abs(prev_err - err) < 1e-14 and it > 5:
                break
            prev_err = err
        
        # Check if we found a valid decomposition
        T_rec = _tensor_from(U_curr, V_curr, W_curr)
        if np.allclose(T_rec, T, atol=1e-8, rtol=1e-8):
            U_r = _round_half(U_curr)
            V_r = _round_half(V_curr)
            W_r = _round_half(W_curr)
            if np.allclose(_tensor_from(U_r, V_r, W_r), T, atol=1e-8, rtol=1e-8):
                return U_r, V_r, W_r
            return U_curr, V_curr, W_curr
    
    return None

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    T = _build_target_tensor()
    
    U_base, V_base, W_base = _build_strassen_decomposition()
    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = U_base.shape[1]
    
    T_rec = _tensor_from(best_U, best_V, best_W)
    if not np.allclose(T_rec, T, atol=1e-6, rtol=1e-6):
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_sb
    
    # Try to reduce rank
    for r in [49, 48, 47, 46, 45, 44, 43, 42, 41, 40]:
        if r >= best_rank:
            continue
        if _time_left() < 10:
            break
        
        n_trials = 20 if r >= 48 else 10
        max_iter = 50
        
        result = _alt_minimize_fast(best_U, best_V, best_W, T, n_trials=n_trials, max_iter=max_iter, r=r)
        if result is not None:
            U_new, V_new, W_new = result
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = r
    
    if _time_left() > 15:
        for r in range(best_rank - 1, max(best_rank - 15, 30), -1):
            if _time_left() < 5:
                break
            if r >= best_rank:
                continue
            result = _alt_minimize_fast(best_U, best_V, best_W, T, n_trials=10, max_iter=50, r=r)
            if result is not None:
                U_new, V_new, W_new = result
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = r
                break
    
    T_rec = _tensor_from(best_U, best_V, best_W)
    if not np.allclose(T_rec, T, atol=1e-6, rtol=1e-6):
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
