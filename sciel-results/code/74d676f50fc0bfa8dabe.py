"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import numpy as np
import time

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


def _schoolbook(ni, mi, pi):
    """Schoolbook decomposition for ni x mi x pi. Returns U, V, W."""
    rank = ni * mi * pi
    U = np.zeros((ni * mi, rank))
    V = np.zeros((mi * pi, rank))
    W = np.zeros((pi * ni, rank))
    col = 0
    for i in range(ni):
        for j in range(mi):
            for k in range(pi):
                U[i * mi + j, col] = 1
                V[j * pi + k, col] = 1
                W[k * ni + i, col] = 1
                col += 1
    return U, V, W


def _embed_block(U_local, V_local, W_local, n, m, p, n_off, m_off, p_off, ni, mi, pi):
    """
    Embed a local decomposition of size ni x mi x pi into global dimensions.
    Returns global U, V, W slices (as arrays of shape (n*m, r), etc.)
    """
    r = U_local.shape[1]
    U_g = np.zeros((n * m, r))
    V_g = np.zeros((m * p, r))
    W_g = np.zeros((p * n, r))
    
    # U: local index li*mi + lj -> global (n_off+li)*m + (m_off+lj)
    for li in range(ni):
        for lj in range(mi):
            local_idx = li * mi + lj
            global_idx = (n_off + li) * m + (m_off + lj)
            U_g[global_idx, :] = U_local[local_idx, :]
    
    # V: local index lj*pi + lk -> global (m_off+lj)*p + (p_off+lk)
    for lj in range(mi):
        for lk in range(pi):
            local_idx = lj * pi + lk
            global_idx = (m_off + lj) * p + (p_off + lk)
            V_g[global_idx, :] = V_local[local_idx, :]
    
    # W: local index lk*ni + li -> global (p_off+lk)*n + (n_off+li)
    for lk in range(pi):
        for li in range(ni):
            local_idx = lk * ni + li
            global_idx = (p_off + lk) * n + (n_off + li)
            W_g[global_idx, :] = W_local[local_idx, :]
    
    return U_g, V_g, W_g


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Recursive block decomposition with Strassen for 2x2x2 blocks
    def _decompose_block(ni, mi, pi, n_off, m_off, p_off):
        """Recursively decompose a ni x mi x pi block. Returns list of (U, V, W, n_off, m_off, p_off, ni, mi, pi)."""
        # Base case: use Strassen for 2x2x2, schoolbook otherwise
        if ni == 2 and mi == 2 and pi == 2:
            U_loc, V_loc, W_loc = _strassen_2x2x2()
            return [(U_loc, V_loc, W_loc, n_off, m_off, p_off, ni, mi, pi)]
        
        # Try to split the largest dimension
        # Split strategy: if a dimension >= 2 and not already 2x2x2, split it
        splits = []
        
        # Try splitting n if ni > 1
        if ni >= 2:
            if ni == 2:
                n_splits = [1, 1]
            elif ni == 3:
                n_splits = [2, 1]
            else:
                n_splits = [ni // 2, ni - ni // 2]
            for idx, n_sub in enumerate(n_splits):
                n_sub_off = n_off + sum(n_splits[:idx])
                sub_blocks = _decompose_block(n_sub, mi, pi, n_sub_off, m_off, p_off)
                splits.extend(sub_blocks)
            if splits:
                return splits
        
        # Try splitting m if mi > 1
        if mi >= 2:
            if mi == 2:
                m_splits = [1, 1]
            elif mi == 3:
                m_splits = [2, 1]
            else:
                m_splits = [mi // 2, mi - mi // 2]
            for idx, m_sub in enumerate(m_splits):
                m_sub_off = m_off + sum(m_splits[:idx])
                sub_blocks = _decompose_block(ni, m_sub, pi, n_off, m_sub_off, p_off)
                splits.extend(sub_blocks)
            if splits:
                return splits
        
        # Try splitting p if pi > 1
        if pi >= 2:
            if pi == 2:
                p_splits = [1, 1]
            elif pi == 3:
                p_splits = [2, 1]
            else:
                p_splits = [pi // 2, pi - pi // 2]
            for idx, p_sub in enumerate(p_splits):
                p_sub_off = p_off + sum(p_splits[:idx])
                sub_blocks = _decompose_block(ni, mi, p_sub, n_off, m_off, p_sub_off)
                splits.extend(sub_blocks)
            if splits:
                return splits
        
        # Base case: schoolbook
        U_loc, V_loc, W_loc = _schoolbook(ni, mi, pi)
        return [(U_loc, V_loc, W_loc, n_off, m_off, p_off, ni, mi, pi)]
    
    # Get all sub-blocks
    all_blocks = _decompose_block(n, m, p, 0, 0, 0)
    
    # Calculate total rank
    total_rank = sum(U.shape[1] for U, V, W, *_ in all_blocks)
    
    # Construct global U, V, W
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for U_loc, V_loc, W_loc, n_off, m_off, p_off, ni, mi, pi in all_blocks:
        r = U_loc.shape[1]
        U_g, V_g, W_g = _embed_block(
            U_loc, V_loc, W_loc, n, m, p,
            n_off, m_off, p_off,
            ni, mi, pi
        )
        
        U[:, col:col+r] += U_g
        V[:, col:col+r] += V_g
        W[:, col:col+r] += W_g
        col += r
    
    # Verify the decomposition is exact
    if time.time() < deadline:
        # Build target tensor
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        # Reconstruct and check using matrix multiplication for speed
        # C[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        # = sum_t U[a,t] * (V[b,t] * W[c,t])
        # Compute VW first: (m*p, p*n) matrix where VW[b,c] = sum_t V[b,t]*W[c,t]
        VW = V @ W.T  # shape (m*p, p*n)
        # Then C[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        # This is not simply U @ VW because of the contraction over t
        # C = np.einsum('at,bc->abc', U, VW) won't work directly
        # Actually: C[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        # Let's compute it as: for each t, outer product
        # Or use: C = np.einsum('at,bt,ct->abc', U, V, W)
        C = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
        
        max_err = np.max(np.abs(C - T))
        if max_err > 1e-9:
            # Fallback to schoolbook if verification fails
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=np.float64)
            V = np.zeros((m * p, rank), dtype=np.float64)
            W = np.zeros((p * n, rank), dtype=np.float64)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1
    
    def to_list(mat):
        return [[float(mat[i, j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

# EVOLVE-BLOCK-END
