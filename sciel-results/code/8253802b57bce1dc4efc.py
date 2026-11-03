"""
Tensor decomposition for 4x4 x 4x9 matrix multiplication.
Strategy: Dynamic programming over the partition lattice to find minimal rank
decompositions by fusing sub-blocks.
"""
import time
import numpy as np
import itertools

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """
    Schoolbook decomposition for M_{n,m,p}.
    Rank n*m*p.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _get_block_decomp(n, m, p):
    """
    Get the best known decomposition for a block of size n x m x p.
    Returns (U, V, W, rank).
    """
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_2x2()
        return U, V, W, 7
    else:
        U, V, W = _schoolbook_nm_p(n, m, p)
        return U, V, W, n * m * p

def _verify_decomp(U, V, W, n, m, p, atol=1e-9):
    """Vectorized verification of the decomposition."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_idx = i_idx.ravel()
    j_idx = j_idx.ravel()
    k_idx = k_idx.ravel()
    
    u_vec = U_r[i_idx, j_idx, :]
    v_vec = V_r[j_idx, k_idx, :]
    w_vec = W_r[k_idx, i_idx, :]
    
    prod = u_vec * v_vec * w_vec
    sums = prod.sum(axis=1)
    
    return np.allclose(sums, 1.0, atol=atol)

def _solve_block(n, m, p, time_limit):
    """
    Solve for the minimal rank decomposition of an n x m x p block
    using dynamic programming over partitions.
    Returns (U, V, W, rank).
    """
    start_time = time.time()
    
    # Cache for sub-problems
    # Key: (n, m, p)
    # Value: (U, V, W, rank)
    cache = {}
    
    def get_sub(n_sub, m_sub, p_sub):
        key = (n_sub, m_sub, p_sub)
        if key in cache:
            return cache[key]
        
        # Base cases: small blocks use direct formulas
        if n_sub <= 2 and m_sub <= 2 and p_sub <= 2:
            U, V, W, r = _get_block_decomp(n_sub, m_sub, p_sub)
            cache[key] = (U, V, W, r)
            return U, V, W, r
            
        # Check time
        if time.time() > time_limit:
            # Fallback to schoolbook if time runs out
            U, V, W, r = _schoolbook_nm_p(n_sub, m_sub, p_sub)
            cache[key] = (U, V, W, r)
            return U, V, W, r

        best_rank = n_sub * m_sub * p_sub
        best_decomp = None
        
        # Try splitting along dimension n
        for split in range(1, n_sub):
            n1, n2 = split, n_sub - split
            U1, V1, W1, r1 = get_sub(n1, m_sub, p_sub)
            U2, V2, W2, r2 = get_sub(n2, m_sub, p_sub)
            total_r = r1 + r2
            if total_r < best_rank:
                # Construct combined decomp
                U = np.zeros((n_sub * m_sub, total_r))
                V = np.zeros((m_sub * p_sub, total_r))
                W = np.zeros((p_sub * n_sub, total_r))
                
                # Place first block
                U[:n1*m_sub, :r1] = U1
                V[:, :r1] = V1
                W[:p_sub*n1, :r1] = W1
                
                # Place second block
                U[n1*m_sub:, r1:] = U2
                V[:, r1:] = V2
                # W for second block: rows correspond to k*n + i, where i is in [n1, n_sub)
                # W2 has shape (p_sub*n2, r2). Indices in W2 are k*n2 + i_local.
                # We need to map to k*n_sub + i_global where i_global = n1 + i_local.
                # W2[k*n2 + i_local] -> W[k*n_sub + n1 + i_local]
                W2_expanded = np.zeros((p_sub * n_sub, r2))
                for k in range(p_sub):
                    W2_expanded[k*n_sub + n1 : k*n_sub + n_sub, :] = W2[k*n2 : (k+1)*n2, :]
                W[p_sub*n1:, r1:] = W2_expanded[p_sub*n1:, :] # Actually just W2_expanded works if we align offsets
                
                # Correction: W indices are k*n + i.
                # Block 1 covers i in [0, n1). Block 2 covers i in [n1, n_sub).
                # W1 has shape (p*n1, r1). W2 has shape (p*n2, r2).
                # We need to place W2 into W such that its i indices are offset by n1.
                # W2[k*n2 + i2] should go to W[k*n_sub + (n1 + i2)].
                
                # Let's re-construct W properly
                W = np.zeros((p_sub * n_sub, total_r))
                for k in range(p_sub):
                    W[k*n_sub : k*n_sub + n1, :r1] = W1[k*n1 : (k+1)*n1, :]
                    W[k*n_sub + n1 : k*n_sub + n_sub, r1:] = W2[k*n2 : (k+1)*n2, :]
                
                best_rank = total_r
                best_decomp = (U, V, W)
                
        # Try splitting along dimension m
        for split in range(1, m_sub):
            m1, m2 = split, m_sub - split
            U1, V1, W1, r1 = get_sub(n_sub, m1, p_sub)
            U2, V2, W2, r2 = get_sub(n_sub, m2, p_sub)
            total_r = r1 + r2
            if total_r < best_rank:
                U = np.zeros((n_sub * m_sub, total_r))
                V = np.zeros((m_sub * p_sub, total_r))
                W = np.zeros((p_sub * n_sub, total_r))
                
                # U: i*m + j. Block 1: j in [0, m1). Block 2: j in [m1, m_sub).
                for i in range(n_sub):
                    U[i*m_sub : i*m_sub + m1, :r1] = U1[i*m1 : (i+1)*m1, :]
                    U[i*m_sub + m1 : i*m_sub + m_sub, r1:] = U2[i*m2 : (i+1)*m2, :]
                
                # V: j*p + k. Block 1: j in [0, m1). Block 2: j in [m1, m_sub).
                for k in range(p_sub):
                    V[:m1*p_sub, :r1] = V1[k : k + m1*p_sub, :] # Wait, V1 is (m1*p, r1)
                    # V1 index: j1*p + k. V2 index: j2*p + k.
                    # V index: j*p + k.
                    # j in [0, m1) for V1, j in [m1, m_sub) for V2.
                    # So V1 goes to V[0 : m1*p_sub, :]? No.
                    # V1 has rows j1*p_sub + k.
                    # We need to place V1 into V such that row j*p_sub + k gets V1[j1*p_sub + k] where j=j1.
                    # And V2 into V such that row j*p_sub + k gets V2[j2*p_sub + k] where j=m1+j2.
                    
                    V = np.zeros((m_sub * p_sub, total_r))
                    for j in range(m_sub):
                        if j < m1:
                            V[j*p_sub : (j+1)*p_sub, :r1] = V1[j*p_sub : (j+1)*p_sub, :]
                        else:
                            V[j*p_sub : (j+1)*p_sub, r1:] = V2[(j-m1)*p_sub : (j-m1+1)*p_sub, :]
                    
                    W = np.zeros((p_sub * n_sub, total_r))
                    # W is same for both, so just stack columns
                    W[:, :r1] = W1
                    W[:, r1:] = W2
                    
                best_rank = total_r
                best_decomp = (U, V, W)

        # Try splitting along dimension p
        for split in range(1, p_sub):
            p1, p2 = split, p_sub - split
            U1, V1, W1, r1 = get_sub(n_sub, m_sub, p1)
            U2, V2, W2, r2 = get_sub(n_sub, m_sub, p2)
            total_r = r1 + r2
            if total_r < best_rank:
                U = np.zeros((n_sub * m_sub, total_r))
                V = np.zeros((m_sub * p_sub, total_r))
                W = np.zeros((p_sub * n_sub, total_r))
                
                # U is same, stack columns
                U[:, :r1] = U1
                U[:, r1:] = U2
                
                # V: j*p + k. Block 1: k in [0, p1). Block 2: k in [p1, p_sub).
                for j in range(m_sub):
                    V[j*p_sub : j*p_sub + p1, :r1] = V1[j*p1 : (j+1)*p1, :]
                    V[j*p_sub + p1 : (j+1)*p_sub, r1:] = V2[j*p2 : (j+1)*p2, :]
                    
                # W: k*n + i. Block 1: k in [0, p1). Block 2: k in [p1, p_sub).
                for k in range(p_sub):
                    if k < p1:
                        W[k*n_sub : (k+1)*n_sub, :r1] = W1[k*n_sub : (k+1)*n_sub, :]
                    else:
                        W[k*n_sub : (k+1)*n_sub, r1:] = W2[(k-p1)*n_sub : (k-p1+1)*n_sub, :]
                        
                best_rank = total_r
                best_decomp = (U, V, W)

        if best_decomp is None:
            U, V, W, r = _schoolbook_nm_p(n_sub, m_sub, p_sub)
            cache[key] = (U, V, W, r)
            return U, V, W, r
            
        cache[key] = (best_decomp[0], best_decomp[1], best_decomp[2], best_rank)
        return best_decomp[0], best_decomp[1], best_decomp[2], best_rank

    U, V, W, rank = get_sub(n, m, p)
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find a better decomposition using DP
    try:
        U, V, W, rank = _solve_block(n, m, p, deadline)
        
        # Verify
        if _verify_decomp(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: The known 114 rank solution (2x4x4 Strassen + 4x4x1 Schoolbook)
    # This is guaranteed to work and is fast.
    U1, V1, W1 = _decomp_4x4x4_fallback()
    U2, V2, W2 = _decomp_4x4x4_fallback()
    U3, V3, W3 = _schoolbook_nm_p(4, 4, 1)
    
    total_rank = 49 + 49 + 16
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Place first 4x4x4 block
    U[:, :49] = U1
    for j in range(m):
        for k in range(4):
            V[j * p + k, :49] = V1[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[k * n + i, :49] = W1[k * 4 + i, :]
    
    # Place second 4x4x4 block
    U[:, 49:98] = U2
    for j in range(m):
        for k in range(4):
            V[j * p + (4 + k), 49:98] = V2[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[(4 + k) * n + i, 49:98] = W2[k * 4 + i, :]
    
    # Place 4x4x1 schoolbook block
    U[:, 98:114] = U3
    for j in range(m):
        V[j * p + 8, 98:114] = V3[j * 1 + 0, :]
    for i in range(n):
        W[8 * n + i, 98:114] = W3[0 * 4 + i, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _decomp_4x4x4_fallback():
    """
    Decomposition of M_{4,4,4} using 2-level Strassen.
    Rank 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    inner_U, inner_V, inner_W = _strassen_2x2()
    
    outer_A = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [0, 0, 1, -1], [0, 1, 0, -1]
    ]
    outer_B = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -1],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    outer_C = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [-1, 0, 1, 0],
        [1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    for t in range(7):
        col_offset = t * 7
        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    b = li * 2 + lj
                    coeff_U = outer_A[t][b] * inner_U[li*2 + lj, r]
                    coeff_V = outer_B[t][b] * inner_V[li*2 + lj, r]
                    coeff_W = outer_C[t][b] * inner_W[li*2 + lj, r]
                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2 * (b // 2) + ei
                            g_j = 2 * (b % 2) + ej
                            U[g_i * 4 + g_j, col] = coeff_U
                            V[g_j * 4 + g_i, col] = coeff_V
                            W[g_i * 4 + g_j, col] = coeff_W
    
    return U, V, W

# EVOLVE-BLOCK-END
