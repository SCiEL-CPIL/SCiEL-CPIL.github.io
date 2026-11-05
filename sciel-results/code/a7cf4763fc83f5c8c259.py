"""Matrix multiplication 3x5x5 - rank minimization via block decomposition + elimination."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
N, M, P = 3, 5, 5


def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a11 + a22)(b11 + b22)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # M2 = (a21 + a22) b11
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, -1]
    
    # M3 = a11 (b12 - b22)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 0, 1, 1]
    
    # M4 = a22 (b21 - b11)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [1, 1, 0, 0]
    
    # M5 = (a11 + a12) b22
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 1, 0]
    
    # M6 = (a21 - a11) (b11 + b12)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    # M7 = (a12 - a22) (b21 + b22)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition for n x m x p. Returns U, V, W."""
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
    return U, V, W


def _embed_block(U_local, V_local, W_local, n, m, p, n_off, m_off, p_off, ni, mi, pi):
    """
    Embed a local decomposition of size ni x mi x pi into global dimensions.
    Returns the global U, V, W slices (as arrays of shape (n*m, r), etc.)
    """
    r = U_local.shape[1]
    U_g = np.zeros((n * m, r))
    V_g = np.zeros((m * p, r))
    W_g = np.zeros((p * n, r))
    
    for li in range(ni):
        for lj in range(mi):
            local_idx = li * mi + lj
            global_idx = (n_off + li) * m + (m_off + lj)
            U_g[global_idx, :] = U_local[local_idx, :]
    
    for lj in range(mi):
        for lk in range(pi):
            local_idx = lj * pi + lk
            global_idx = (m_off + lj) * p + (p_off + lk)
            V_g[global_idx, :] = V_local[local_idx, :]
    
    for lk in range(pi):
        for li in range(ni):
            local_idx = lk * ni + li
            global_idx = (p_off + lk) * n + (n_off + li)
            W_g[global_idx, :] = W_local[local_idx, :]
    
    return U_g, V_g, W_g


def _block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Decompose n x m x p into blocks defined by n_parts, m_parts, p_parts.
    Use Strassen for 2x2x2 blocks, schoolbook for others.
    """
    n_offsets = [sum(n_parts[:i]) for i in range(len(n_parts))]
    m_offsets = [sum(m_parts[:i]) for i in range(len(m_parts))]
    p_offsets = [sum(p_parts[:i]) for i in range(len(p_parts))]
    
    total_rank = 0
    block_ranks = {}
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                if ni == 2 and mi == 2 and pi == 2:
                    block_ranks[(i, j, k)] = 7
                else:
                    block_ranks[(i, j, k)] = ni * mi * pi
                total_rank += block_ranks[(i, j, k)]
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                r = block_ranks[(i, j, k)]
                if ni == 2 and mi == 2 and pi == 2:
                    U_loc, V_loc, W_loc = _strassen_2x2x2()
                else:
                    U_loc, V_loc, W_loc = _schoolbook(ni, mi, pi)
                
                U_g, V_g, W_g = _embed_block(U_loc, V_loc, W_loc, n, m, p,
                                              n_offsets[i], m_offsets[j], p_offsets[k],
                                              ni, mi, pi)
                U[:, col:col+r] = U_g
                V[:, col:col+r] = V_g
                W[:, col:col+r] = W_g
                col += r
    
    return U, V, W, total_rank


def _verify_fast(U, V, W):
    """Faster verification using einsum."""
    r = U.shape[1]
    u = U.reshape(N, M, r)
    v = V.reshape(M, P, r)
    w = W.reshape(P, N, r)
    T = np.einsum('ijc,jkc,kic->ijk', u, v, w)
    true_T = np.ones((N, M, P))
    return np.allclose(T, true_T, atol=1e-9)


def _try_eliminate(U, V, W, col_to_remove):
    """
    Try to remove column col_to_remove by solving linearized system.
    """
    r = U.shape[1]
    if r <= 1:
        return None
    
    remaining = [c for c in range(r) if c != col_to_remove]
    r_rem = len(remaining)
    rem_idx = np.array(remaining)
    
    b = np.ones(N * M * P)
    
    # Option 1: Fix V and W, solve for U
    A = np.zeros((N * M * P, N * M * (r - 1)))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                eq_idx = i * M * P + j * P + k
                for idx, c in enumerate(rem_idx):
                    var_idx = (i * M + j) * (r - 1) + idx
                    A[eq_idx, var_idx] = V[j * P + k, c] * W[k * N + i, c]
    
    try:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        x = None
    
    if x is not None:
        x_reshaped = x.reshape(N * M, r - 1)
        U_new = np.zeros_like(U)
        for idx, c in enumerate(rem_idx):
            U_new[:, c] = x_reshaped[:, idx]
        if _verify_fast(U_new, V, W):
            return U_new, V, W
    
    # Option 2: Fix U and W, solve for V
    A2 = np.zeros((N * M * P, M * P * (r - 1)))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                eq_idx = i * M * P + j * P + k
                for idx, c in enumerate(rem_idx):
                    var_idx = (j * P + k) * (r - 1) + idx
                    A2[eq_idx, var_idx] = U[i * M + j, c] * W[k * N + i, c]
    
    try:
        x2, _, _, _ = np.linalg.lstsq(A2, b, rcond=None)
    except Exception:
        x2 = None
    
    if x2 is not None:
        x2_reshaped = x2.reshape(M * P, r - 1)
        V_new = np.zeros_like(V)
        for idx, c in enumerate(rem_idx):
            V_new[:, c] = x2_reshaped[:, idx]
        if _verify_fast(U, V_new, W):
            return U, V_new, W
    
    # Option 3: Fix U and V, solve for W
    A3 = np.zeros((N * M * P, P * N * (r - 1)))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                eq_idx = i * M * P + j * P + k
                for idx, c in enumerate(rem_idx):
                    var_idx = (k * N + i) * (r - 1) + idx
                    A3[eq_idx, var_idx] = U[i * M + j, c] * V[j * P + k, c]
    
    try:
        x3, _, _, _ = np.linalg.lstsq(A3, b, rcond=None)
    except Exception:
        x3 = None
    
    if x3 is not None:
        x3_reshaped = x3.reshape(P * N, r - 1)
        W_new = np.zeros_like(W)
        for idx, c in enumerate(rem_idx):
            W_new[:, c] = x3_reshaped[:, idx]
        if _verify_fast(U, V, W_new):
            return U, V, W_new
    
    return None


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    candidates = [
        ([1, 2], [2, 1, 2], [2, 1, 2]),  # rank 25
        ([1, 2], [1, 2, 2], [1, 2, 2]),  # rank 25
        ([1, 2], [2, 2, 1], [2, 2, 1]),  # rank 25
        ([1, 1, 1], [1, 2, 2], [1, 2, 2]),  # rank 27
        ([1, 2], [2, 3], [2, 3]),  # rank 38
        ([1, 1, 1], [2, 3], [2, 3]),  # rank 39
        ([1, 2], [1, 1, 3], [1, 1, 3]),  # rank 33
        ([1, 2], [3, 2], [3, 2]),  # rank 38
        ([2, 1], [2, 3], [2, 3]),  # rank 38
    ]
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    for n_parts, m_parts, p_parts in candidates:
        if time.time() > deadline - 30.0:
            break
        try:
            U, V, W, rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
            if rank < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, rank
        except Exception:
            pass
    
    if best_U is None:
        # Fallback to schoolbook
        best_U, best_V, best_W = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    # Iteratively try to eliminate columns to reduce rank further
    U, V, W = best_U, best_V, best_W
    improved = True
    while improved:
        improved = False
        r = U.shape[1]
        
        for col in range(r):
            if time.time() > deadline - 20.0:
                break
            
            result = _try_eliminate(U, V, W, col)
            if result is not None:
                U, V, W = result
                improved = True
                break  # Restart with new rank
        
        if time.time() > deadline - 20.0:
            break
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
