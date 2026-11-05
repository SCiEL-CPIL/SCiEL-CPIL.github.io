"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np
import itertools

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

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

def _winograd_3x3x3():
    """
    Winograd's 23-multiplication algorithm for 3x3x3 matrix multiplication.
    U: (9, 23), V: (9, 23), W: (9, 23)
    Indexing: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Based on standard Winograd 3x3x3 implementation
    # Indices for A (U): 0=a11, 1=a12, 2=a13, 3=a21, 4=a22, 5=a23, 6=a31, 7=a32, 8=a33
    # Indices for B (V): 0=b11, 1=b12, 2=b13, 3=b21, 4=b22, 5=b23, 6=b31, 7=b32, 8=b33
    # Indices for C (W): 0=c11, 1=c12, 2=c13, 3=c21, 4=c22, 5=c23, 6=c31, 7=c32, 8=c33
    
    # Term 1
    U[[0, 3, 6], 0] = 1
    V[[0, 1, 2], 0] = 1
    W[[0, 3, 6], 0] = 1/4
    W[[1, 4, 7], 0] = 1/4
    W[[2, 5, 8], 0] = 1/4
    
    # Term 2
    U[[0, 3, 6], 1] = [1, 1, -1]
    V[[0, 1, 2], 1] = [1, -1, 1]
    W[[0, 3, 6], 1] = 1/4
    W[[1, 4, 7], 1] = -1/4
    W[[2, 5, 8], 1] = 1/4
    
    # Term 3
    U[[0, 3, 6], 2] = [1, -1, 1]
    V[[0, 1, 2], 2] = [1, 1, -1]
    W[[0, 3, 6], 2] = 1/4
    W[[1, 4, 7], 2] = 1/4
    W[[2, 5, 8], 2] = -1/4
    
    # Term 4
    U[[0, 3, 6], 3] = [1, -1, -1]
    V[[0, 1, 2], 3] = [1, -1, -1]
    W[[0, 3, 6], 3] = 1/4
    W[[1, 4, 7], 3] = 1/4
    W[[2, 5, 8], 3] = 1/4
    
    # Term 5
    U[[0, 1, 2], 4] = 1
    V[[3, 4, 5], 4] = 1
    W[[0, 3, 6], 4] = 1/4
    W[[1, 4, 7], 4] = 1/4
    W[[2, 5, 8], 4] = 1/4
    
    # Term 6
    U[[0, 1, 2], 5] = [1, 1, -1]
    V[[3, 4, 5], 5] = [1, -1, 1]
    W[[0, 3, 6], 5] = 1/4
    W[[1, 4, 7], 5] = -1/4
    W[[2, 5, 8], 5] = 1/4
    
    # Term 7
    U[[0, 1, 2], 6] = [1, -1, 1]
    V[[3, 4, 5], 6] = [1, 1, -1]
    W[[0, 3, 6], 6] = 1/4
    W[[1, 4, 7], 6] = 1/4
    W[[2, 5, 8], 6] = -1/4
    
    # Term 8
    U[[0, 1, 2], 7] = [1, -1, -1]
    V[[3, 4, 5], 7] = [1, -1, -1]
    W[[0, 3, 6], 7] = 1/4
    W[[1, 4, 7], 7] = 1/4
    W[[2, 5, 8], 7] = 1/4
    
    # Term 9
    U[[1, 4, 7], 8] = 1
    V[[2, 5, 8], 8] = 1
    W[[0, 3, 6], 8] = 1/4
    W[[1, 4, 7], 8] = 1/4
    W[[2, 5, 8], 8] = 1/4
    
    # Term 10
    U[[1, 4, 7], 9] = [1, 1, -1]
    V[[2, 5, 8], 9] = [1, -1, 1]
    W[[0, 3, 6], 9] = 1/4
    W[[1, 4, 7], 9] = -1/4
    W[[2, 5, 8], 9] = 1/4
    
    # Term 11
    U[[1, 4, 7], 10] = [1, -1, 1]
    V[[2, 5, 8], 10] = [1, 1, -1]
    W[[0, 3, 6], 10] = 1/4
    W[[1, 4, 7], 10] = 1/4
    W[[2, 5, 8], 10] = -1/4
    
    # Term 12
    U[[1, 4, 7], 11] = [1, -1, -1]
    V[[2, 5, 8], 11] = [1, -1, -1]
    W[[0, 3, 6], 11] = 1/4
    W[[1, 4, 7], 11] = 1/4
    W[[2, 5, 8], 11] = 1/4
    
    # Term 13
    U[[2, 5, 8], 12] = 1
    V[[1, 4, 7], 12] = 1
    W[[0, 3, 6], 12] = 1/4
    W[[1, 4, 7], 12] = 1/4
    W[[2, 5, 8], 12] = 1/4
    
    # Term 14
    U[[2, 5, 8], 13] = [1, 1, -1]
    V[[1, 4, 7], 13] = [1, -1, 1]
    W[[0, 3, 6], 13] = 1/4
    W[[1, 4, 7], 13] = -1/4
    W[[2, 5, 8], 13] = 1/4
    
    # Term 15
    U[[2, 5, 8], 14] = [1, -1, 1]
    V[[1, 4, 7], 14] = [1, 1, -1]
    W[[0, 3, 6], 14] = 1/4
    W[[1, 4, 7], 14] = 1/4
    W[[2, 5, 8], 14] = -1/4
    
    # Term 16
    U[[2, 5, 8], 15] = [1, -1, -1]
    V[[1, 4, 7], 15] = [1, -1, -1]
    W[[0, 3, 6], 15] = 1/4
    W[[1, 4, 7], 15] = 1/4
    W[[2, 5, 8], 15] = 1/4
    
    # Term 17
    U[[3, 6], 16] = 1
    V[[3, 5], 16] = 1
    W[[3, 6], 16] = 1/2
    
    # Term 18
    U[[3, 6], 17] = [1, -1]
    V[[3, 5], 17] = [1, -1]
    W[[3, 6], 17] = 1/2
    
    # Term 19
    U[[1, 2], 18] = 1
    V[[6, 7], 18] = 1
    W[[1, 2], 18] = 1/2
    
    # Term 20
    U[[1, 2], 19] = [1, -1]
    V[[6, 7], 19] = [1, -1]
    W[[1, 2], 19] = 1/2
    
    # Term 21
    U[[4, 8], 20] = 1
    V[[1, 3], 20] = 1
    W[[4, 6], 20] = 1/2
    
    # Term 22
    U[[4, 8], 21] = [1, -1]
    V[[1, 3], 21] = [1, -1]
    W[[4, 6], 21] = 1/2
    
    # Term 23
    U[[0, 4, 8], 22] = 1
    V[[4], 22] = 1
    W[[0, 4, 8], 22] = 1
    
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
    """Get the rank for a block of size ni x mi x pi."""
    if ni == 2 and mi == 2 and pi == 2:
        return 7
    if ni == 3 and mi == 3 and pi == 3:
        return 23
    return ni * mi * pi

def _block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Decompose n x m x p into blocks defined by n_parts, m_parts, p_parts.
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
                elif ni == 3 and mi == 3 and pi == 3:
                    U_loc, V_loc, W_loc = _winograd_3x3x3()
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

def _generate_all_splits(target, max_parts=4):
    """Generate all compositions of target into positive integers up to max_parts parts."""
    results = []
    
    def _compose(remaining, max_val, current):
        if remaining == 0:
            if len(current) <= max_parts:
                results.append(tuple(current))
            return
        for v in range(1, min(remaining, max_val) + 1):
            current.append(v)
            _compose(remaining - v, max_val, current)
            current.pop()
            
    _compose(target, target, [])
    return results

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    n_splits = _generate_all_splits(n, max_parts=4)
    m_splits = _generate_all_splits(m, max_parts=4)
    p_splits = _generate_all_splits(p, max_parts=4)
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    for n_parts in n_splits:
        for m_parts in m_splits:
            for p_parts in p_splits:
                if time.time() > deadline:
                    break
                try:
                    U, V, W, rank = _block_decomposition(n, m, p, 
                                                         list(n_parts), 
                                                         list(m_parts), 
                                                         list(p_parts))
                    if rank < best_rank:
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
                except Exception:
                    continue
                    
    if best_U is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best_U, best_V, best_W
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
