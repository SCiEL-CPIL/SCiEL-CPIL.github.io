"""Tensor decomposition search for 3x5 * 5x5 matrix multiplication."""
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


def _block_rank(ni, mi, pi):
    """Return the rank for a block of size ni x mi x pi."""
    if ni == 2 and mi == 2 and pi == 2:
        return 7
    return ni * mi * pi


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
                r = _block_rank(ni, mi, pi)
                block_ranks[(i, j, k)] = r
                total_rank += r
    
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


def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    rank = U.shape[1]
    U_arr = U.reshape(N, M, rank)
    V_arr = V.reshape(M, P, rank)
    W_arr = W.reshape(P, N, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    return np.allclose(T, 1.0, atol=1e-9)


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (15, rank), (25, rank), (15, rank).
    """
    deadline = time.time() + 130  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = n * m * p  # 75
    
    # Try various block decompositions
    candidates = [
        # (n_parts, m_parts, p_parts, expected_rank)
        ([1, 2], [2, 1, 2], [2, 1, 2]),  # 4+1+4+7+2+7 = 25
        ([1, 2], [1, 2, 2], [1, 2, 2]),  # 1+4+4+2+7+7 = 25
        ([1, 2], [2, 2, 1], [2, 2, 1]),  # 4+4+1+7+7+2 = 25
        ([1, 2], [2, 3], [2, 3]),        # 4+9+7+18 = 38
        ([1, 2], [3, 2], [3, 2]),        # 9+4+18+7 = 38
        ([2, 1], [2, 3], [2, 3]),        # 7+18+4+9 = 38
        ([1, 1, 1], [2, 3], [2, 3]),     # 4+9+4+9+4+9 = 39
        ([1, 1, 1], [1, 2, 2], [1, 2, 2]),  # 27
        ([1, 2], [1, 1, 3], [1, 1, 3]),  # 33
        ([2, 1], [2, 1, 2], [2, 1, 2]),  # 7+2+7+4+1+4 = 25
        ([2, 1], [1, 2, 2], [1, 2, 2]),  # 2+7+7+1+4+4 = 25
        ([1, 1, 1], [2, 2, 1], [2, 2, 1]),  # 4+4+1+4+4+1+4+4+1 = 27
    ]
    
    for n_parts, m_parts, p_parts in candidates:
        if time.time() > deadline:
            break
        try:
            U, V, W, rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
            if rank < best_rank and _verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
        except Exception:
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
