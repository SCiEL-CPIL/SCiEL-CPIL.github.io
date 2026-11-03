"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _get_strassen_222():
    """
    Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7).
    """
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
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
    """
    Returns U, V, W for 2x2x1 multiplication using schoolbook (Rank 4).
    """
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """
    Fast verification using numpy.
    """
    r = U.shape[1]
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    
    for i in range(n):
        M = np.zeros((m, m))
        for k in range(p):
            B = U_r[i, :, :] * W_r[k, i, :]
            Vk = V_r[:, k, :]
            M += B @ Vk.T
        
        if not np.allclose(M, np.eye(m), atol=1e-9):
            return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, deadline=None):
    """
    Try to reduce rank by zeroing out columns.
    """
    if deadline is None:
        deadline = time.time() + 5
    
    r_orig = U.shape[1]
    
    # Try zeroing out one column at a time
    for col_to_zero in range(r_orig):
        if time.time() > deadline:
            break
        
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        U_test[:, col_to_zero] = 0
        V_test[:, col_to_zero] = 0
        W_test[:, col_to_zero] = 0
        
        if _verify_fast(U_test, V_test, W_test, n, m, p):
            mask = [c for c in range(r_orig) if c != col_to_zero]
            U_red = U_test[:, mask]
            V_red = V_test[:, mask]
            W_red = W_test[:, mask]
            return U_red, V_red, W_red
    
    return U, V, W

def _construct_initial(n, m, p):
    """
    Construct initial Strassen-block decomposition (Rank 50).
    """
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [
        (0, 2, 'strassen'),
        (2, 4, 'strassen'),
        (4, 6, 'strassen'),
        (6, 7, 'school')
    ]
    
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    
    for j_start, j_end in m_blocks:
        for k_start, k_end, p_type in p_blocks:
            if p_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
                p_sub = 2
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
                p_sub = 1
            
            for i in range(n):
                for j_sub in range(j_end - j_start):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            
            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            
            col_offset += r_sub
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with Rank 50 Strassen-block
    U, V, W = _construct_initial(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    if not _verify_fast(U, V, W, n, m, p):
        # Fallback to Schoolbook if construction failed (should not happen)
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Randomized Search: Rescale columns and try to reduce rank
    # We use the property: u' = lambda u, v' = lambda v, w' = w / lambda^2
    # This preserves the tensor sum.
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        r = U.shape[1]
        
        # Generate random scaling factors
        # Avoid very small or very large values to maintain numerical stability
        lambdas = rng.uniform(0.1, 10.0, size=r)
        
        # Apply scaling
        U = U * lambdas
        V = V * lambdas
        W = W / (lambdas ** 2)
        
        # Try to reduce rank
        U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p, deadline=deadline)
        
        if U_red.shape[1] < r:
            U, V, W = U_red, V_red, W_red
            best_rank = U.shape[1]
            best_U, best_V, best_W = U, V, W
            # If we reduced rank, we can restart the search with the new lower rank
            # Or just continue with the current one.
            # Restarting might be better to explore the new manifold.
            # But for now, just continue.
            
        # If we get stuck, we might try a more aggressive mix?
        # For now, simple rescaling is a good start.
        
        # Check time
        if time.time() > deadline - 10:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
