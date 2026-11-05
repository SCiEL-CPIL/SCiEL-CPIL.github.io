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
    
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, -1]
    
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 0, 1, 1]
    
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [1, 1, 0, 0]
    
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 1, 0]
    
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
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
    """Embed a local decomposition of size ni x mi x pi into global dimensions."""
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


def _recursive_block_rank(ni, mi, pi):
    """Compute the optimal rank for a block using recursive Strassen splitting."""
    if ni == 0 or mi == 0 or pi == 0:
        return 0
    if ni == 1 and mi == 1 and pi == 1:
        return 1
    if ni == 1 and mi == 1:
        return pi
    if ni == 1 and pi == 1:
        return mi
    if mi == 1 and pi == 1:
        return ni
    if ni == 1:
        return mi * pi
    if mi == 1:
        return ni * pi
    if pi == 1:
        return ni * mi
    
    # Try all possible splits and pick the minimum rank
    best = ni * mi * pi  # schoolbook fallback
    
    # Try splitting n dimension
    if ni >= 2:
        for a in range(1, ni):
            r = _recursive_block_rank(a, mi, pi) + _recursive_block_rank(ni - a, mi, pi)
            if r < best:
                best = r
    # Try splitting m dimension
    if mi >= 2:
        for b in range(1, mi):
            r = _recursive_block_rank(ni, b, pi) + _recursive_block_rank(ni, mi - b, pi)
            if r < best:
                best = r
    # Try splitting p dimension
    if pi >= 2:
        for c in range(1, pi):
            r = _recursive_block_rank(ni, mi, c) + _recursive_block_rank(ni, mi, pi - c)
            if r < best:
                best = r
    
    # Special case: 2x2x2 uses Strassen (rank 7)
    if ni == 2 and mi == 2 and pi == 2:
        best = min(best, 7)
    
    return best


def _recursive_block_decomp(ni, mi, pi):
    """
    Recursively decompose ni x mi x pi using optimal splits.
    Returns U (ni*mi, rank), V (mi*pi, rank), W (pi*ni, rank).
    """
    if ni == 0 or mi == 0 or pi == 0:
        rank = 0
        return np.zeros((ni*mi, rank)), np.zeros((mi*pi, rank)), np.zeros((pi*ni, rank))
    if ni == 1 and mi == 1 and pi == 1:
        U = np.array([[1.0]])
        V = np.array([[1.0]])
        W = np.array([[1.0]])
        return U, V, W
    if ni == 1 and mi == 1:
        return _schoolbook(ni, mi, pi)
    if ni == 1 and pi == 1:
        return _schoolbook(ni, mi, pi)
    if mi == 1 and pi == 1:
        return _schoolbook(ni, mi, pi)
    if ni == 1:
        return _schoolbook(ni, mi, pi)
    if mi == 1:
        return _schoolbook(ni, mi, pi)
    if pi == 1:
        return _schoolbook(ni, mi, pi)
    
    # Special case: 2x2x2 uses Strassen
    if ni == 2 and mi == 2 and pi == 2:
        return _strassen_2x2x2()
    
    # Try all splits and pick the one with minimum rank
    best_rank = ni * mi * pi
    best_result = None
    
    # Try splitting n dimension
    for a in range(1, ni):
        b = ni - a
        r1 = _recursive_block_rank(a, mi, pi)
        r2 = _recursive_block_rank(b, mi, pi)
        total_r = r1 + r2
        if total_r < best_rank:
            best_rank = total_r
            U1, V1, W1 = _recursive_block_decomp(a, mi, pi)
            U2, V2, W2 = _recursive_block_decomp(b, mi, pi)
            # Embed U1 at rows [0, a*mi), U2 at rows [a*mi, ni*mi)
            U = np.zeros((ni*mi, total_r))
            V = np.zeros((mi*pi, total_r))
            W = np.zeros((pi*ni, total_r))
            U[:a*mi, :r1] = U1
            U[a*mi:, r1:r1+r2] = U2
            V[:, :r1] = V1
            V[:, r1:r1+r2] = V2
            W[:pi*a, :r1] = W1
            W[pi*a:, r1:r1+r2] = W2
            best_result = (U, V, W)
    
    # Try splitting m dimension
    for b in range(1, mi):
        a = mi - b
        r1 = _recursive_block_rank(ni, b, pi)
        r2 = _recursive_block_rank(ni, a, pi)
        total_r = r1 + r2
        if total_r < best_rank:
            best_rank = total_r
            U1, V1, W1 = _recursive_block_decomp(ni, b, pi)
            U2, V2, W2 = _recursive_block_decomp(ni, a, pi)
            U = np.zeros((ni*mi, total_r))
            V = np.zeros((mi*pi, total_r))
            W = np.zeros((pi*ni, total_r))
            # U1 covers rows with j in [0,b), U2 covers j in [b,mi)
            for i in range(ni):
                for j in range(b):
                    U[i*mi+j, :r1] = U1[i*b+j, :r1]
                for j in range(b, mi):
                    U[i*mi+j, r1:r1+r2] = U2[i*a+(j-b), r1:r1+r2]
            for j in range(b):
                for k in range(pi):
                    V[j*pi+k, :r1] = V1[j*pi+k, :r1]
            for j in range(b, mi):
                for k in range(pi):
                    V[j*pi+k, r1:r1+r2] = V2[(j-b)*pi+k, r1:r1+r2]
            W[:, :r1] = W1
            W[:, r1:r1+r2] = W2
            best_result = (U, V, W)
    
    # Try splitting p dimension
    for c in range(1, pi):
        d = pi - c
        r1 = _recursive_block_rank(ni, mi, c)
        r2 = _recursive_block_rank(ni, mi, d)
        total_r = r1 + r2
        if total_r < best_rank:
            best_rank = total_r
            U1, V1, W1 = _recursive_block_decomp(ni, mi, c)
            U2, V2, W2 = _recursive_block_decomp(ni, mi, d)
            U = np.zeros((ni*mi, total_r))
            V = np.zeros((mi*pi, total_r))
            W = np.zeros((pi*ni, total_r))
            U[:, :r1] = U1
            U[:, r1:r1+r2] = U2
            for j in range(mi):
                for k in range(c):
                    V[j*pi+k, :r1] = V1[j*c+k, :r1]
                for k in range(c, pi):
                    V[j*pi+k, r1:r1+r2] = V2[j*d+(k-c), r1:r1+r2]
            for k in range(c):
                for i in range(ni):
                    W[k*ni+i, :r1] = W1[k*ni+i, :r1]
            for k in range(c, pi):
                for i in range(ni):
                    W[k*ni+i, r1:r1+r2] = W2[(k-c)*ni+i, r1:r1+r2]
            best_result = (U, V, W)
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return _schoolbook(ni, mi, pi)


def _block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Decompose n x m x p into blocks defined by n_parts, m_parts, p_parts.
    Each block uses recursive decomposition.
    """
    n_offsets = [sum(n_parts[:i]) for i in range(len(n_parts))]
    m_offsets = [sum(m_parts[:i]) for i in range(len(m_parts))]
    p_offsets = [sum(p_parts[:i]) for i in range(len(p_parts))]
    
    total_rank = 0
    block_ranks = {}
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                r = _recursive_block_rank(ni, mi, pi)
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
                U_loc, V_loc, W_loc = _recursive_block_decomp(ni, mi, pi)
                
                U_g, V_g, W_g = _embed_block(U_loc, V_loc, W_loc, n, m, p,
                                              n_offsets[i], m_offsets[j], p_offsets[k],
                                              ni, mi, pi)
                U[:, col:col+r] = U_g
                V[:, col:col+r] = V_g
                W[:, col:col+r] = W_g
                col += r
    
    return U, V, W, total_rank


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using tensor contraction via einsum."""
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # For each (i,j,k), we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Reshape for einsum:
    U_r = U.reshape(n, m, -1)  # (n, m, rank)
    V_r = V.reshape(m, p, -1)  # (m, p, rank)
    W_r = W.reshape(p, n, -1)  # (p, n, rank)
    
    # Compute the full tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    # Check if all entries are 1
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True


def _generate_splits(dim, max_parts=5):
    """Generate all compositions of dim into up to max_parts positive integers."""
    splits = []
    def helper(remaining, parts_so_far):
        if remaining == 0:
            if parts_so_far:
                splits.append(list(parts_so_far))
            return
        if len(parts_so_far) >= max_parts:
            return
        for next_val in range(1, remaining + 1):
            helper(remaining - next_val, parts_so_far + [next_val])
    helper(dim, [])
    return splits


def _estimate_rank(n_parts, m_parts, p_parts):
    """Estimate the total rank for a given split using recursive ranks."""
    total = 0
    for ni in n_parts:
        for mi in m_parts:
            for pi in p_parts:
                total += _recursive_block_rank(ni, mi, pi)
    return total


def _row_sharing_decomposition(n, m, p):
    """
    Row-sharing decomposition: treat the n x m x p tensor as n independent
    1 x m x p sub-tensors (row-vector times matrix).
    For each row i, we need to compute (row_i of A) * B.
    The rank of 1 x m x p is m*p (schoolbook, since it's just vector-matrix mult).
    Total rank = n * m * p... but wait, that's the same as schoolbook.
    
    Actually, the key insight is that for 1 x m x p, the rank is exactly m*p
    (you can't do better than schoolbook for a single row).
    
    However, we can potentially share structure between rows if we use a
    smarter approach. Let's just use the block decomposition with n_parts = [1,1,1].
    """
    # This is equivalent to block decomposition with n split into single rows
    n_parts = [1] * n
    m_parts = [m]
    p_parts = [p]
    return _block_decomposition(n, m, p, n_parts, m_parts, p_parts)


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Get the best block decomposition using recursive Strassen
    n_splits = _generate_splits(n, max_parts=3)
    m_splits = _generate_splits(m, max_parts=5)
    p_splits = _generate_splits(p, max_parts=5)
    
    all_candidates = []
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                rank = _estimate_rank(ns, ms, ps)
                all_candidates.append((rank, ns, ms, ps))
    
    all_candidates.sort(key=lambda x: x[0])
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    # Try candidates in order of estimated rank
    for rank_est, n_parts, m_parts, p_parts in all_candidates:
        if time.time() > deadline:
            break
        if rank_est >= best_rank:
            continue
        
        try:
            U, V, W, actual_rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
            if actual_rank < best_rank:
                if _verify_decomposition_fast(U, V, W, n, m, p):
                    best_rank = actual_rank
                    best_U, best_V, best_W = U, V, W
        except Exception:
            continue
    
    # Fallback to schoolbook if something went wrong
    if best_U is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
