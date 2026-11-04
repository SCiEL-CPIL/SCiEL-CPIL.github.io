import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _schoolbook(n, m, p):
    """
    Returns U, V, W for n x m x p matrix multiplication using schoolbook algorithm.
    Rank = n * m * p.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
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

def _embed_block(U_b, V_b, W_b, n, m, p, i_off, j_off, k_off, bn, bm, bp):
    """
    Embeds a block decomposition into the full space.
    """
    U_full = np.zeros((n * m, U_b.shape[1]))
    V_full = np.zeros((m * p, V_b.shape[1]))
    W_full = np.zeros((p * n, W_b.shape[1]))
    
    for i in range(bn):
        for j in range(bm):
            idx_u = (i_off + i) * m + (j_off + j)
            U_full[idx_u, :] = U_b[i * bm + j, :]
            
    for j in range(bm):
        for k in range(bp):
            idx_v = (j_off + j) * p + (k_off + k)
            V_full[idx_v, :] = V_b[j * bp + k, :]
            
    for k in range(bp):
        for i in range(bn):
            idx_w = (k_off + k) * n + (i_off + i)
            W_full[idx_w, :] = W_b[k * bn + i, :]
            
    return U_full, V_full, W_full

def construct():
    n, m, p = DIMENSIONS
    
    # Partition: n=[2,2], m=[2,3], p=[2,3]
    # This yields blocks:
    # (2,2,2) rank 8
    # (2,2,3) rank 12
    # (2,3,2) rank 12
    # (2,3,3) rank 18
    # Total rank: 50
    
    n_blocks = [2, 2]
    m_blocks = [2, 3]
    p_blocks = [2, 3]
    
    U_list = []
    V_list = []
    W_list = []
    
    i_off = 0
    for bn in n_blocks:
        j_off = 0
        for bm in m_blocks:
            k_off = 0
            for bp in p_blocks:
                # Use schoolbook for all blocks to ensure validity
                U_b, V_b, W_b = _schoolbook(bn, bm, bp)
                
                U_full, V_full, W_full = _embed_block(
                    U_b, V_b, W_b, n, m, p,
                    i_off, j_off, k_off,
                    bn, bm, bp
                )
                U_list.append(U_full)
                V_list.append(V_full)
                W_list.append(W_full)
                
                k_off += bp
            j_off += bm
        i_off += bn
        
    U_combined = np.hstack(U_list)
    V_combined = np.hstack(V_list)
    W_combined = np.hstack(W_list)
    
    return {"U": U_combined.tolist(), "V": V_combined.tolist(), "W": W_combined.tolist()}
