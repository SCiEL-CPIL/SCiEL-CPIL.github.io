import numpy as np
import time

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] = 1 if c = k*n+i where a=i*m+j, b=j*p+k"""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _als_decompose(T, rank, max_iter=500, n_restarts=200, seed=42):
    """
    Alternating Least Squares to find a rank decomposition of tensor T.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    n_a, n_b, n_c = T.shape
    rng = np.random.RandomState(seed)
    best_error = np.inf
    best_result = None
    
    for restart in range(n_restarts):
        U = rng.randn(n_a, rank) * 0.1
        V = rng.randn(n_b, rank) * 0.1
        W = rng.randn(n_c, rank) * 0.1
        
        for it in range(max_iter):
            # Solve for U: T[a, b, c] = U[a,r] * (V[b,r]*W[c,r])
            # Reshape T to (n_a, n_b*n_c)
            T_reshaped = T.reshape(n_a, n_b * n_c)
            # Z[b*n_c+c, r] = V[b,r] * W[c,r]
            Z = np.zeros((n_b * n_c, rank))
            for b in range(n_b):
                for c in range(n_c):
                    Z[b * n_c + c, :] = V[b, :] * W[c, :]
            # U = T_reshaped @ pinv(Z.T)
            try:
                U_new = T_reshaped @ np.linalg.pinv(Z.T)
            except:
                break
            
            # Solve for V
            T_reshaped_v = np.zeros((n_b, n_a * n_c))
            for b in range(n_b):
                for a in range(n_a):
                    for c in range(n_c):
                        T_reshaped_v[b, a * n_c + c] = T[a, b, c]
            Z2 = np.zeros((n_a * n_c, rank))
            for a in range(n_a):
                for c in range(n_c):
                    Z2[a * n_c + c, :] = U_new[a, :] * W[c, :]
            try:
                V_new = T_reshaped_v @ np.linalg.pinv(Z2.T)
            except:
                break
            
            # Solve for W
            T_reshaped_w = np.zeros((n_c, n_a * n_b))
            for c in range(n_c):
                for a in range(n_a):
                    for b in range(n_b):
                        T_reshaped_w[c, a * n_b + b] = T[a, b, c]
            Z3 = np.zeros((n_a * n_b, rank))
            for a in range(n_a):
                for b in range(n_b):
                    Z3[a * n_b + b, :] = U_new[a, :] * V_new[b, :]
            try:
                W_new = T_reshaped_w @ np.linalg.pinv(Z3.T)
            except:
                break
            
            U, V, W = U_new, V_new, W_new
        
        # Check error
        T_check = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.max(np.abs(T_check - T))
        
        if error < best_error:
            best_error = error
            best_result = (U.copy(), V.copy(), W.copy())
        
        if error < 1e-10:
            break
    
    return best_result, best_error

def _try_round(U, V, W, T, scale=1):
    """Try rounding to integers or half-integers and verify."""
    # Try integers
    U_r = np.round(U)
    V_r = np.round(V)
    W_r = np.round(W)
    T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
    if np.max(np.abs(T_check - T)) < 1e-9:
        return (U_r.astype(np.int64), V_r.astype(np.int64), W_r.astype(np.int64))
    
    # Try half-integers
    U_h = np.round(U * 2) / 2
    V_h = np.round(V * 2) / 2
    W_h = np.round(W * 2) / 2
    T_check = np.einsum('ar,br,cr->abc', U_h, V_h, W_h)
    if np.max(np.abs(T_check - T)) < 1e-9:
        return (U_h, V_h, W_h)
    
    # Try scaled integers
    for s in [1, 2, 3]:
        U_s = np.round(U * s) / s
        V_s = np.round(V * s) / s
        W_s = np.round(W * s) / s
        T_check = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
        if np.max(np.abs(T_check - T)) < 1e-9:
            return (U_s, V_s, W_s)
    
    return None

def _get_2x3x4_decomposition(max_time=60):
    """
    Find a rank-10 integer/half-integer decomposition for 2x3x4 matrix multiplication.
    """
    n, m, p = 2, 3, 4
    rank = 10
    T = _build_target_tensor(n, m, p)
    
    start = time.time()
    best = None
    
    # Try ALS with various seeds
    for seed in range(0, 1000):
        if time.time() - start > max_time - 5:
            break
        result, error = _als_decompose(T, rank, max_iter=200, n_restarts=1, seed=seed)
        if result is not None and error < 1e-8:
            rounded = _try_round(*result, T)
            if rounded is not None:
                best = rounded
                break
    
    return best

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    T_check = np.einsum('ad,bd,cd->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_exp[a_idx, b_idx, c_idx] = 1.0
    return np.max(np.abs(T_check - T_exp)) < 1e-9

def _build_rank44_fallback(n, m, p):
    """Build the rank-44 Strassen-based decomposition as fallback."""
    rank = 44
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    col = 0
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Standard terms for A[:,0] * B[0,:]
        U[0*m + 0, col] = 1; V[0*p + k0, col] = 1; W[k0*n + 0, col] = 1; col += 1
        U[0*m + 0, col] = 1; V[0*p + k1, col] = 1; W[k1*n + 0, col] = 1; col += 1
        U[1*m + 0, col] = 1; V[0*p + k0, col] = 1; W[k0*n + 1, col] = 1; col += 1
        U[1*m + 0, col] = 1; V[0*p + k1, col] = 1; W[k1*n + 1, col] = 1; col += 1
        
        # Strassen terms for the 2x2 sub-block (A[:,1:3] * B[1:3,:])
        U[0*m + 1, col] = 1; U[1*m + 2, col] = 1; V[1*p + k0, col] = 1; V[2*p + k1, col] = 1; W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1; col += 1
        U[1*m + 1, col] = 1; U[1*m + 2, col] = 1; V[1*p + k0, col] = 1; W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1; col += 1
        U[0*m + 1, col] = 1; V[1*p + k1, col] = 1; V[2*p + k1, col] = -1; W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1; col += 1
        U[1*m + 2, col] = 1; V[2*p + k0, col] = 1; V[1*p + k0, col] = -1; W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1; col += 1
        U[0*m + 1, col] = 1; U[0*m + 2, col] = 1; V[2*p + k1, col] = 1; W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1; col += 1
        U[1*m + 1, col] = 1; U[0*m + 1, col] = -1; V[1*p + k0, col] = 1; V[1*p + k1, col] = 1; W[k1*n + 1, col] = 1; col += 1
        U[0*m + 2, col] = 1; U[1*m + 2, col] = -1; V[2*p + k0, col] = 1; V[2*p + k1, col] = 1; W[k0*n + 0, col] = 1; col += 1
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.
    Strategy: Split 8 columns into two 4-column blocks, use rank-10 decomposition for 2x3x4.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 2, 3, 8
    
    # Try to find rank-10 decomposition for 2x3x4
    decomp_2x3x4 = None
    if time.time() < deadline - 30:
        decomp_2x3x4 = _get_2x3x4_decomposition(max_time=80)
    
    if decomp_2x3x4 is not None:
        U_block, V_block, W_block = decomp_2x3x4
        rank = 2 * 10
        U = np.zeros((n*m, rank), dtype=U_block.dtype if hasattr(U_block, 'dtype') else np.int64)
        V = np.zeros((m*p, rank), dtype=V_block.dtype if hasattr(V_block, 'dtype') else np.int64)
        W = np.zeros((p*n, rank), dtype=W_block.dtype if hasattr(W_block, 'dtype') else np.int64)
        
        for block in range(2):
            k0 = block * 4
            col_start = block * 10
            col_end = col_start + 10
            
            U[:, col_start:col_end] = U_block
            
            for j in range(3):
                for kk in range(4):
                    V[j * p + k0 + kk, col_start:col_end] = V_block[j * 4 + kk, :]
            
            for kk in range(4):
                for i in range(2):
                    W[(k0 + kk) * n + i, col_start:col_end] = W_block[kk * 2 + i, :]
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-44
    U, V, W = _build_rank44_fallback(n, m, p)
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Ultimate fallback: rank-48 schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb), dtype=np.int64)
    V_sb = np.zeros((m*p, rank_sb), dtype=np.int64)
    W_sb = np.zeros((p*n, rank_sb), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U_sb[a_idx, c] = 1
                V_sb[b_idx, c] = 1
                W_sb[c_idx, c] = 1
                c += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
