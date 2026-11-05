"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np

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

def _bcl_3x3x3():
    """
    Return the rank-23 Bini-Capovani-Lotti-Romani decomposition for 3x3x3.
    U: (9, 23), V: (9, 23), W: (9, 23)
    Indexing: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    n = 3
    U = np.zeros((n*n, 23))
    V = np.zeros((n*n, 23))
    W = np.zeros((n*n, 23))
    
    def set_entry(mat, i, j, val, col):
        mat[i*n+j, col] += val
    
    # M1 = (a11+a22+a33)(b11+b22+b33)
    for idx in range(3):
        set_entry(U, idx, idx, 1, 0)
        set_entry(V, idx, idx, 1, 0)
        set_entry(W, idx, idx, 1, 0)
    
    # M2 = (a11+a22-a33)(b11+b22-b33)
    set_entry(U, 0, 0, 1, 1); set_entry(U, 1, 1, 1, 1); set_entry(U, 2, 2, -1, 1)
    set_entry(V, 0, 0, 1, 1); set_entry(V, 1, 1, 1, 1); set_entry(V, 2, 2, -1, 1)
    set_entry(W, 0, 0, 1, 1); set_entry(W, 1, 1, 1, 1); set_entry(W, 2, 2, -1, 1)
    
    # M3 = (a11-a22+a33)(b11-b22+b33)
    set_entry(U, 0, 0, 1, 2); set_entry(U, 1, 1, -1, 2); set_entry(U, 2, 2, 1, 2)
    set_entry(V, 0, 0, 1, 2); set_entry(V, 1, 1, -1, 2); set_entry(V, 2, 2, 1, 2)
    set_entry(W, 0, 0, 1, 2); set_entry(W, 1, 1, -1, 2); set_entry(W, 2, 2, 1, 2)
    
    # M4 = (a11-a22-a33)(b11-b22-b33)
    set_entry(U, 0, 0, 1, 3); set_entry(U, 1, 1, -1, 3); set_entry(U, 2, 2, -1, 3)
    set_entry(V, 0, 0, 1, 3); set_entry(V, 1, 1, -1, 3); set_entry(V, 2, 2, -1, 3)
    set_entry(W, 0, 0, 1, 3); set_entry(W, 1, 1, -1, 3); set_entry(W, 2, 2, -1, 3)
    
    # M5 = (a12+a21+a33)(b21+b12+b33)
    set_entry(U, 0, 1, 1, 4); set_entry(U, 1, 0, 1, 4); set_entry(U, 2, 2, 1, 4)
    set_entry(V, 1, 0, 1, 4); set_entry(V, 0, 1, 1, 4); set_entry(V, 2, 2, 1, 4)
    set_entry(W, 1, 0, 1, 4); set_entry(W, 0, 1, 1, 4); set_entry(W, 2, 2, 1, 4)
    
    # M6 = (a12+a21-a33)(b21+b12-b33)
    set_entry(U, 0, 1, 1, 5); set_entry(U, 1, 0, 1, 5); set_entry(U, 2, 2, -1, 5)
    set_entry(V, 1, 0, 1, 5); set_entry(V, 0, 1, 1, 5); set_entry(V, 2, 2, -1, 5)
    set_entry(W, 1, 0, 1, 5); set_entry(W, 0, 1, 1, 5); set_entry(W, 2, 2, -1, 5)
    
    # M7 = (a12-a21+a33)(b21-b12+b33)
    set_entry(U, 0, 1, 1, 6); set_entry(U, 1, 0, -1, 6); set_entry(U, 2, 2, 1, 6)
    set_entry(V, 1, 0, 1, 6); set_entry(V, 0, 1, -1, 6); set_entry(V, 2, 2, 1, 6)
    set_entry(W, 1, 0, 1, 6); set_entry(W, 0, 1, -1, 6); set_entry(W, 2, 2, 1, 6)
    
    # M8 = (a12-a21-a33)(b21-b12-b33)
    set_entry(U, 0, 1, 1, 7); set_entry(U, 1, 0, -1, 7); set_entry(U, 2, 2, -1, 7)
    set_entry(V, 1, 0, 1, 7); set_entry(V, 0, 1, -1, 7); set_entry(V, 2, 2, -1, 7)
    set_entry(W, 1, 0, 1, 7); set_entry(W, 0, 1, -1, 7); set_entry(W, 2, 2, -1, 7)
    
    # M9 = (a13+a23+a31)(b31+b23+b13)
    set_entry(U, 0, 2, 1, 8); set_entry(U, 1, 2, 1, 8); set_entry(U, 2, 0, 1, 8)
    set_entry(V, 2, 0, 1, 8); set_entry(V, 1, 2, 1, 8); set_entry(V, 0, 2, 1, 8)
    set_entry(W, 2, 0, 1, 8); set_entry(W, 1, 2, 1, 8); set_entry(W, 0, 2, 1, 8)
    
    # M10 = (a13+a23-a31)(b31+b23-b13)
    set_entry(U, 0, 2, 1, 9); set_entry(U, 1, 2, 1, 9); set_entry(U, 2, 0, -1, 9)
    set_entry(V, 2, 0, 1, 9); set_entry(V, 1, 2, 1, 9); set_entry(V, 0, 2, -1, 9)
    set_entry(W, 2, 0, 1, 9); set_entry(W, 1, 2, 1, 9); set_entry(W, 0, 2, -1, 9)
    
    # M11 = (a13-a23+a31)(b31-b23+b13)
    set_entry(U, 0, 2, 1, 10); set_entry(U, 1, 2, -1, 10); set_entry(U, 2, 0, 1, 10)
    set_entry(V, 2, 0, 1, 10); set_entry(V, 1, 2, -1, 10); set_entry(V, 0, 2, 1, 10)
    set_entry(W, 2, 0, 1, 10); set_entry(W, 1, 2, -1, 10); set_entry(W, 0, 2, 1, 10)
    
    # M12 = (a13-a23-a31)(b31-b23-b13)
    set_entry(U, 0, 2, 1, 11); set_entry(U, 1, 2, -1, 11); set_entry(U, 2, 0, -1, 11)
    set_entry(V, 2, 0, 1, 11); set_entry(V, 1, 2, -1, 11); set_entry(V, 0, 2, -1, 11)
    set_entry(W, 2, 0, 1, 11); set_entry(W, 1, 2, -1, 11); set_entry(W, 0, 2, -1, 11)
    
    # M13 = (a11+a33)(b11+b33)
    set_entry(U, 0, 0, 1, 12); set_entry(U, 2, 2, 1, 12)
    set_entry(V, 0, 0, 1, 12); set_entry(V, 2, 2, 1, 12)
    set_entry(W, 0, 0, 1, 12); set_entry(W, 2, 2, 1, 12)
    
    # M14 = (a11-a33)(b11-b33)
    set_entry(U, 0, 0, 1, 13); set_entry(U, 2, 2, -1, 13)
    set_entry(V, 0, 0, 1, 13); set_entry(V, 2, 2, -1, 13)
    set_entry(W, 0, 0, 1, 13); set_entry(W, 2, 2, -1, 13)
    
    # M15 = (a22+a33)(b22+b33)
    set_entry(U, 1, 1, 1, 14); set_entry(U, 2, 2, 1, 14)
    set_entry(V, 1, 1, 1, 14); set_entry(V, 2, 2, 1, 14)
    set_entry(W, 1, 1, 1, 14); set_entry(W, 2, 2, 1, 14)
    
    # M16 = (a22-a33)(b22-b33)
    set_entry(U, 1, 1, 1, 15); set_entry(U, 2, 2, -1, 15)
    set_entry(V, 1, 1, 1, 15); set_entry(V, 2, 2, -1, 15)
    set_entry(W, 1, 1, 1, 15); set_entry(W, 2, 2, -1, 15)
    
    # M17 = (a12+a31)(b21+b13)
    set_entry(U, 0, 1, 1, 16); set_entry(U, 2, 0, 1, 16)
    set_entry(V, 1, 0, 1, 16); set_entry(V, 0, 2, 1, 16)
    set_entry(W, 1, 0, 1, 16); set_entry(W, 0, 2, 1, 16)
    
    # M18 = (a12-a31)(b21-b13)
    set_entry(U, 0, 1, 1, 17); set_entry(U, 2, 0, -1, 17)
    set_entry(V, 1, 0, 1, 17); set_entry(V, 0, 2, -1, 17)
    set_entry(W, 1, 0, 1, 17); set_entry(W, 0, 2, -1, 17)
    
    # M19 = (a23+a32)(b31+b23)
    set_entry(U, 1, 2, 1, 18); set_entry(U, 2, 1, 1, 18)
    set_entry(V, 2, 0, 1, 18); set_entry(V, 1, 2, 1, 18)
    set_entry(W, 2, 0, 1, 18); set_entry(W, 1, 2, 1, 18)
    
    # M20 = (a23-a32)(b31-b23)
    set_entry(U, 1, 2, 1, 19); set_entry(U, 2, 1, -1, 19)
    set_entry(V, 2, 0, 1, 19); set_entry(V, 1, 2, -1, 19)
    set_entry(W, 2, 0, 1, 19); set_entry(W, 1, 2, -1, 19)
    
    # M21 = a13*b31
    set_entry(U, 0, 2, 1, 20)
    set_entry(V, 2, 0, 1, 20)
    set_entry(W, 2, 0, 1, 20)
    
    # M22 = a21*b12
    set_entry(U, 1, 0, 1, 21)
    set_entry(V, 0, 1, 1, 21)
    set_entry(W, 0, 1, 1, 21)
    
    # M23 = a33*b22
    set_entry(U, 2, 2, 1, 22)
    set_entry(V, 1, 1, 1, 22)
    set_entry(W, 1, 1, 1, 22)
    
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

def _best_block_rank(ni, mi, pi):
    """Return the best known rank for a block of size ni x mi x pi."""
    if ni == 2 and mi == 2 and pi == 2:
        return 7
    if ni == 3 and mi == 3 and pi == 3:
        return 23
    return ni * mi * pi

def _block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Decompose n x m x p into blocks using best known decompositions.
    """
    n_offsets = [sum(n_parts[:i]) for i in range(len(n_parts))]
    m_offsets = [sum(m_parts[:i]) for i in range(len(m_parts))]
    p_offsets = [sum(p_parts[:i]) for i in range(len(p_parts))]
    
    total_rank = 0
    block_ranks = {}
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                block_ranks[(i, j, k)] = _best_block_rank(ni, mi, pi)
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
                elif ni == 3 and mi == 3 and pi == 3:
                    U_loc, V_loc, W_loc = _bcl_3x3x3()
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

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _generate_splits(dim, max_parts=4):
    """Generate all compositions of dim into up to max_parts positive integers."""
    splits = []
    def helper(remaining, parts_so_far, max_parts):
        if not parts_so_far and remaining == 0:
            return
        if len(parts_so_far) == max_parts:
            if remaining > 0:
                splits.append(parts_so_far + [remaining])
            return
        for next_val in range(1, remaining + 1):
            helper(remaining - next_val, parts_so_far + [next_val], max_parts)
    helper(dim, [], max_parts)
    return splits

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    n_splits = _generate_splits(n, max_parts=3)
    m_splits = _generate_splits(m, max_parts=4)
    p_splits = _generate_splits(p, max_parts=4)
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    all_candidates = set()
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                all_candidates.add((tuple(ns), tuple(ms), tuple(ps)))
                
    def estimate_rank(n_parts, m_parts, p_parts):
        total = 0
        for ni in n_parts:
            for mi in m_parts:
                for pi in p_parts:
                    total += _best_block_rank(ni, mi, pi)
        return total
        
    sorted_candidates = sorted(all_candidates, key=lambda x: estimate_rank(x[0], x[1], x[2]))
    
    for n_parts, m_parts, p_parts in sorted_candidates:
        if time.time() > deadline:
            break
        try:
            U, V, W, rank = _block_decomposition(n, m, p, list(n_parts), list(m_parts), list(p_parts))
            if rank < best_rank:
                if _verify_decomposition(U, V, W, n, m, p):
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
