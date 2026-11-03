"""Matrix multiplication 2x4 x 4x7 via Strassen block decomposition + vectorized alternating minimization."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]
DEADLINE = 135.0

def _time_left():
    return DEADLINE - time.time()

def _tensor_from(U, V, W):
    return np.einsum('ar,br,cr->abc', U, V, W)

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
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    for j_off in [0, 2]:
        for k_off, r_sub, U_sub, V_sub, W_sub in [(0, 7, U_s, V_s, W_s), (2, 7, U_s, V_s, W_s), (4, 7, U_s, V_s, W_s), (6, 4, U_sc, V_sc, W_sc)]:
            p_sub = 2 if r_sub == 7 else 1
            for i in range(2):
                for j in range(2):
                    U[i * 4 + (j_off + j), col_offset:col_offset + r_sub] = U_sub[i * 2 + j, :]
            for j in range(2):
                for k in range(p_sub):
                    V[(j_off + j) * 7 + (k_off + k), col_offset:col_offset + r_sub] = V_sub[j * p_sub + k, :]
            for k in range(p_sub):
                for i in range(2):
                    W[(k_off + k) * 2 + i, col_offset:col_offset + r_sub] = W_sub[k * 2 + i, :]
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

def _alt_minimize_vectorized(U, V, W, T, n_trials=5, max_iter=200, r=None):
    """Vectorized alternating minimization for tensor decomposition."""
    a, b, c = T.shape
    if r is None:
        r = U.shape[1]
    
    # Precompute target tensors in different orderings
    T_abc = T.reshape(a * b * c)
    T_cab = T.transpose(2, 0, 1).reshape(c * a * b)
    T_bac = T.transpose(1, 0, 2).reshape(b * a * c)
    
    for trial in range(n_trials):
        if _time_left() < 3:
            return None
        
        if trial == 0:
            U_curr = U.copy()
            V_curr = V.copy()
            W_curr = W.copy()
        else:
            rng = np.random.RandomState(trial * 137 + 42)
            scale = 0.05 + 0.05 * trial
            U_curr = U + rng.randn(a, r) * scale
            V_curr = V + rng.randn(b, r) * scale
            W_curr = W + rng.randn(c, r) * scale
        
        prev_err = np.inf
        for it in range(max_iter):
            if _time_left() < 2:
                return None
            
            # Optimize W: For each c, T[:,:,c] = sum_d W[c,d] * outer(U[:,d], V[:,d])
            # D_w shape: (c*a*b, r), D_w[(c_val*a + i)*b + j, d] = U[i,d] * V[j,d]
            # Vectorized: D_w = reshape of (U[:,d]*V[:,d] for each d, broadcast over c)
            # D_w[c_val, i, j, d] = U[i,d] * V[j,d]
            UV_outer = np.einsum('id,jd->ijd', U_curr, V_curr)  # (a, b, r)
            D_w = np.tile(UV_outer, (c, 1, 1, 1))  # (c, a, b, r)
            D_w = D_w.reshape(c * a * b, r)
            
            try:
                W_new = np.linalg.lstsq(D_w, T_cab, rcond=None)[0].reshape(c, r)
            except:
                W_new = W_curr
            
            # Optimize U: For each i, T[i,:,:] = sum_d U[i,d] * (V[:,d] * W[:,d])
            # D_u shape: (a*b*c, r), D_u[(i_val*b + j)*c + c_val, d] = V[j,d] * W[c_val,d]
            VW_outer = np.einsum('jd,cd->jcd', V_curr, W_new)  # (b, c, r)
            D_u = np.tile(VW_outer, (a, 1, 1, 1))  # (a, b, c, r)
            D_u = D_u.reshape(a * b * c, r)
            
            try:
                U_new = np.linalg.lstsq(D_u, T_abc, rcond=None)[0].reshape(a, r)
            except:
                U_new = U_curr
            
            # Optimize V: For each j, T[:,j,:] = sum_d V[j,d] * (U[:,d] * W[:,d])
            # D_v shape: (b*a*c, r), D_v[(j_val*a + i)*c + c_val, d] = U[i,d] * W[c_val,d]
            UW_outer = np.einsum('id,cd->icd', U_new, W_new)  # (a, c, r)
            D_v = np.tile(UW_outer, (b, 1, 1, 1))  # (b, a, c, r)
            D_v = D_v.reshape(b * a * c, r)
            
            try:
                V_new = np.linalg.lstsq(D_v, T_bac, rcond=None)[0].reshape(b, r)
            except:
                V_new = V_curr
            
            U_curr, V_curr, W_curr = U_new, V_new, W_new
            
            T_rec = _tensor_from(U_curr, V_curr, W_curr)
            err = np.linalg.norm(T_rec - T)
            if err < 1e-10:
                break
            if abs(prev_err - err) < 1e-14 and it > 20:
                break
            prev_err = err
        
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
    
    # Try rank reduction with vectorized alternating minimization
    # Focus on ranks just below current, with increasing difficulty
    for r in [49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38]:
        if r >= best_rank:
            continue
        if _time_left() < 8:
            break
        
        n_trials = 6 if r >= 47 else 4
        max_iter = 250 if r >= 47 else 150
        
        result = _alt_minimize_vectorized(best_U, best_V, best_W, T, n_trials=n_trials, max_iter=max_iter, r=r)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
    
    # If time remains, try more aggressive lower ranks
    if _time_left() > 12:
        for r in range(best_rank - 1, max(best_rank - 10, 35), -1):
            if _time_left() < 5:
                break
            if r >= best_rank:
                continue
            result = _alt_minimize_vectorized(best_U, best_V, best_W, T, n_trials=4, max_iter=120, r=r)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = r
                break
    
    # Final verification
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
