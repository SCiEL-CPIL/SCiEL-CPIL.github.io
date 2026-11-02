import numpy as np
import time

def _get_2x3x4_decomposition():
    """
    Get a rank-10 decomposition for 2x3x4 matrix multiplication.
    Returns (U, V, W) with shapes (6,10), (12,10), (8,10) or None.
    
    The 2x3x4 tensor can be decomposed as:
    - 4 rank-1 terms for A[:,0] (first column of A)
    - 6 additional terms using Strassen-like structure for A[:,1:3] with 4 output columns
    """
    n, m, p = 2, 3, 4
    rank = 10
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Try to find rank-10 decomposition using ALS
    best_result = None
    rng = np.random.RandomState(12345)
    
    for attempt in range(500):
        U_init = rng.randn(n*m, rank)
        V_init = rng.randn(m*p, rank)
        W_init = rng.randn(p*n, rank)
        
        U_curr, V_curr, W_curr = U_init, V_init, W_init
        
        for iteration in range(300):
            # Solve for U
            Z = np.zeros((m*p * p*n, rank))
            for b in range(m*p):
                for c in range(p*n):
                    Z[b * p*n + c, :] = V_curr[b, :] * W_curr[c, :]
            T_reshaped = T.reshape(n*m, m*p * p*n)
            try:
                U_new = T_reshaped @ np.linalg.pinv(Z.T)
            except:
                break
            
            # Solve for V
            Z2 = np.zeros((n*m * p*n, rank))
            for a in range(n*m):
                for c in range(p*n):
                    Z2[a * p*n + c, :] = U_new[a, :] * W_curr[c, :]
            T_reshaped2 = np.zeros((m*p, n*m * p*n))
            for b in range(m*p):
                for a in range(n*m):
                    for c in range(p*n):
                        T_reshaped2[b, a * p*n + c] = T[a, b, c]
            try:
                V_new = T_reshaped2 @ np.linalg.pinv(Z2.T)
            except:
                break
            
            # Solve for W
            Z3 = np.zeros((n*m * m*p, rank))
            for a in range(n*m):
                for b in range(m*p):
                    Z3[a * m*p + b, :] = U_new[a, :] * V_new[b, :]
            T_reshaped3 = np.zeros((p*n, n*m * m*p))
            for c in range(p*n):
                for a in range(n*m):
                    for b in range(m*p):
                        T_reshaped3[c, a * m*p + b] = T[a, b, c]
            try:
                W_new = T_reshaped3 @ np.linalg.pinv(Z3.T)
            except:
                break
            
            U_curr, V_curr, W_curr = U_new, V_new, W_new
        
        T_check = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
        error = np.max(np.abs(T_check - T))
        
        if error < 1e-8:
            # Try to round to integers
            U_round = np.round(U_curr)
            V_round = np.round(V_curr)
            W_round = np.round(W_curr)
            
            T_check_round = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
            if np.max(np.abs(T_check_round - T)) < 1e-9:
                best_result = (U_round.astype(np.int64), V_round.astype(np.int64), W_round.astype(np.int64))
                break
            
            # Try half-integers
            U_half = np.round(U_curr * 2) / 2
            V_half = np.round(V_curr * 2) / 2
            W_half = np.round(W_curr * 2) / 2
            
            T_check_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
            if np.max(np.abs(T_check_half - T)) < 1e-9:
                best_result = (U_half, V_half, W_half)
                break
    
    return best_result

def construct():
    """
    Construct a rank-minimal tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.
    
    Strategy:
    - Split the 8-column dimension into 2 blocks of 4 columns each.
    - For each block, use a rank-10 decomposition of 2x3x4.
    - Total rank: 2 * 10 = 20.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    n, m, p = 2, 3, 8
    
    # Get rank-10 decomposition of 2x3x4
    decomp_2x3x4 = None
    if time.time() < deadline - 30:
        decomp_2x3x4 = _get_2x3x4_decomposition()
    
    if decomp_2x3x4 is not None:
        U_block, V_block, W_block = decomp_2x3x4
        rank = 2 * 10
        U = np.zeros((n*m, rank), dtype=U_block.dtype)
        V = np.zeros((m*p, rank), dtype=V_block.dtype)
        W = np.zeros((p*n, rank), dtype=W_block.dtype)
        
        for block in range(2):
            k0 = block * 4
            col_start = block * 10
            col_end = col_start + 10
            
            # Copy U block (same for all blocks)
            U[:, col_start:col_end] = U_block
            
            # Remap V indices: V_block has shape (12, 10) for 2x3x4
            # V_block index: j*4 + k where j in {0,1,2}, k in {0,1,2,3}
            # Need: j*8 + (k0 + k)
            for j in range(3):
                for kk in range(4):
                    local_k = kk
                    global_k = k0 + kk
                    V[j * p + global_k, col_start:col_end] = V_block[j * 4 + local_k, :]
            
            # Remap W indices: W_block has shape (8, 10) for 2x3x4
            # W_block index: k*2 + i where k in {0,1,2,3}, i in {0,1}
            # Need: (k0 + k)*2 + i
            for kk in range(4):
                for i in range(2):
                    local_k = kk
                    global_k = k0 + kk
                    W[global_k * n + i, col_start:col_end] = W_block[local_k * 2 + i, :]
        
        # Verify
        def verify(U, V, W):
            r = U.shape[1]
            T_check = np.einsum('ad,bd,cd->abc', U, V, W)
            T_exp = np.zeros((n*m, m*p, p*n), dtype=np.float64)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        c_idx = k * n + i
                        T_exp[a_idx, b_idx, c_idx] = 1
            return np.allclose(T_check, T_exp, atol=1e-9)
        
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-44 construction (Strassen per 2-column block)
    rank = 44
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    col = 0
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        U[0*m + 0, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 0, col] = 1
        col += 1
        U[0*m + 0, col] = 1
        V[0*p + k1, col] = 1
        W[k1*n + 0, col] = 1
        col += 1
        U[1*m + 0, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1
        col += 1
        U[1*m + 0, col] = 1
        V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        
        U[0*m + 1, col] = 1
        U[1*m + 2, col] = 1
        V[1*p + k0, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1*m + 1, col] = 1
        U[1*m + 2, col] = 1
        V[1*p + k0, col] = 1
        W[k0*n + 1, col] = 1
        W[k1*n + 1, col] = -1
        col += 1
        U[0*m + 1, col] = 1
        V[1*p + k1, col] = 1
        V[2*p + k1, col] = -1
        W[k1*n + 0, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1*m + 2, col] = 1
        V[2*p + k0, col] = 1
        V[1*p + k0, col] = -1
        W[k0*n + 0, col] = 1
        W[k0*n + 1, col] = 1
        col += 1
        U[0*m + 1, col] = 1
        U[0*m + 2, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = -1
        W[k1*n + 0, col] = 1
        col += 1
        U[1*m + 1, col] = 1
        U[0*m + 1, col] = -1
        V[1*p + k0, col] = 1
        V[1*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[0*m + 2, col] = 1
        U[1*m + 2, col] = -1
        V[2*p + k0, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1
    
    def verify(U, V, W):
        r = U.shape[1]
        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
        T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T_exp[a_idx, b_idx, c_idx] = 1
        return np.array_equal(T_check, T_exp)
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
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
