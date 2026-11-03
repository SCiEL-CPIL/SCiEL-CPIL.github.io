"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_222():
    """
    Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7).
    """
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Strassen terms
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

def _school_221():
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

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W satisfy the tensor identity exactly using vectorized operations.
    """
    r = U.shape[1]
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    
    # For each (i, j, jp, k), check: sum_r W[k*n+i, r] * U[i*m+j, r] * V[jp*p+k, r] == delta_{j,jp}
    # Vectorize over r using einsum
    # Create index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    jp_idx = np.tile(np.arange(m), n * p)
    k_idx = np.tile(np.arange(p), n * m)
    
    # Reshape for broadcasting
    i_idx = i_idx.reshape(-1, 1)
    j_idx = j_idx.reshape(-1, 1)
    jp_idx = jp_idx.reshape(-1, 1)
    k_idx = k_idx.reshape(-1, 1)
    
    # Gather the needed vectors
    U_vec = U[i_idx * m + j_idx, :]  # (N, r)
    V_vec = V[jp_idx * p + k_idx, :]  # (N, r)
    W_vec = W[k_idx * n + i_idx, :]  # (N, r)
    
    # Compute sum over r
    result = np.einsum('ir,jr,kr->ijk', U_vec, V_vec, W_vec)
    
    # Expected: delta_{j,jp} for each (i, j, jp, k)
    expected = (j_idx.squeeze() == jp_idx.squeeze()).astype(float)
    
    if not np.allclose(result, expected, atol=1e-9):
        return False
    return True

def _build_block_decomposition(n, m, p, m_splits, p_splits):
    """
    Builds a decomposition by splitting m and p into blocks.
    m_splits: list of sizes for m
    p_splits: list of tuples (size, type) where type is 'strassen', 'school', or 'strassen2'
    """
    U_s, V_s, W_s = _strassen_222()
    U_sc, V_sc, W_sc = _school_221()
    
    def _get_sub_decomp(p_size, p_type):
        if p_type == 'strassen':
            return U_s, V_s, W_s, 7
        elif p_type == 'school':
            return U_sc, V_sc, W_sc, 4
        elif p_type == 'strassen2':
            # For p_size=4, use two Strassen 2x2x2 blocks
            r = 14
            U_sub = np.zeros((4, r))
            V_sub = np.zeros((4 * p_size, r))
            W_sub = np.zeros((p_size * 2, r))
            # First 2x2x2 block
            U_sub[:, 0:7] = U_s
            V_sub[:, 0:7] = V_s[:4, :]  # V_s is (4, 7)
            W_sub[:, 0:7] = W_s[:4, :]  # W_s is (4, 7)
            # Second 2x2x2 block
            U_sub[:, 7:14] = U_s
            V_sub[:, 7:14] = V_s[4:8, :]  # Wait, V_s only has 4 rows
            # Actually, for 2x2x4, we need V to have shape (2*4, r) = (8, r)
            # V_s has shape (4, 7), so we need to repeat it
            V_sub[0:4, 7:14] = V_s
            V_sub[4:8, 7:14] = V_s
            W_sub[0:4, 7:14] = W_s
            W_sub[4:8, 7:14] = W_s
            return U_sub, V_sub, W_sub, r
        else:
            raise ValueError(f"Unknown type: {p_type}")
    
    total_rank = 0
    # Calculate total rank first to allocate arrays
    for m_size in m_splits:
        for p_size, p_type in p_splits:
            _, _, _, r_sub = _get_sub_decomp(p_size, p_type)
            total_rank += r_sub * (m_size // 2)
                
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    m_offset = 0
    
    for m_size in m_splits:
        p_offset = 0
        for p_size, p_type in p_splits:
            U_sub, V_sub, W_sub, r_sub = _get_sub_decomp(p_size, p_type)
            num_m_blocks = m_size // 2
            
            for mb in range(num_m_blocks):
                # Place U
                for i in range(n):
                    for j_sub in range(2):
                        j = m_offset + mb * 2 + j_sub
                        U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
                
                # Place V
                for j_sub in range(2):
                    j = m_offset + mb * 2 + j_sub
                    for k_sub in range(p_size):
                        k = p_offset + k_sub
                        V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_size + k_sub, :]
                
                # Place W
                for k_sub in range(p_size):
                    k = p_offset + k_sub
                    for i in range(n):
                        W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
                
                col_offset += r_sub
            
            p_offset += p_size
        m_offset += m_size
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try multiple split strategies and pick the best (lowest rank)
    strategies = [
        ([2, 2], [(2, 'strassen'), (2, 'strassen'), (2, 'strassen'), (1, 'school')]),
        ([4], [(2, 'strassen'), (2, 'strassen'), (2, 'strassen'), (1, 'school')]),
        ([2, 2], [(2, 'strassen'), (2, 'strassen'), (3, 'school')]),
        ([2, 2], [(3, 'school'), (2, 'strassen'), (2, 'strassen')]),
        ([2, 2], [(4, 'strassen2'), (3, 'school')]),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for m_splits, p_splits in strategies:
        if time.time() > deadline:
            break
        try:
            U, V, W = _build_block_decomposition(n, m, p, m_splits, p_splits)
            rank = U.shape[1]
            if rank < best_rank:
                if _verify_decomposition(U, V, W, n, m, p):
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    else:
        # Fallback to trivial decomposition
        r = n * m * p
        U_fb = [[0] * r for _ in range(n * m)]
        V_fb = [[0] * r for _ in range(m * p)]
        W_fb = [[0] * r for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
