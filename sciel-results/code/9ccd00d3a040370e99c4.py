"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # 2-level Strassen decomposition
    # Rank = 7 * 7 = 49
    
    # Define 2x2 Strassen coefficients
    S_A = np.array([
        [1, 0, 0, 1], # l=0
        [0, 0, 1, 1], # l=1
        [1, 0, 0, 0], # l=2
        [0, 0, 0, 1], # l=3
        [1, 1, 0, 0], # l=4
        [-1, 0, 1, 0],# l=5
        [0, 1, 0, -1] # l=6
    ])
    
    S_B = np.array([
        [1, 0, 0, 1], # l=0
        [1, 0, 0, 0], # l=1
        [0, 1, 0, -1],# l=2
        [-1, 0, 1, 0],# l=3
        [0, 0, 0, 1], # l=4
        [1, 1, 0, 0], # l=5
        [0, 0, 1, 1]  # l=6
    ])
    
    S_C = np.array([
        [1, 0, 0, 1], # l=0
        [0, 0, 1, -1],# l=1
        [0, 1, 0, 1], # l=2
        [1, 0, 1, 0], # l=3
        [-1, 1, 0, 0],# l=4
        [0, 0, 0, 1], # l=5
        [1, 0, 0, 0]  # l=6
    ])
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Precompute indices for U, V, W
    i_idx, j_idx = np.indices((n, m))
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    rb_u = (i_idx // 2).flatten()
    cb_u = (j_idx // 2).flatten()
    li_u = (i_idx % 2).flatten()
    lj_u = (j_idx % 2).flatten()
    block_u = rb_u * 2 + cb_u
    local_u = li_u * 2 + lj_u
    flat_u = i_flat * m + j_flat
    
    j_idx, k_idx = np.indices((m, p))
    j_flat_v = j_idx.flatten()
    k_flat_v = k_idx.flatten()
    rb_v = (j_idx // 2).flatten()
    cb_v = (k_idx // 2).flatten()
    li_v = (j_idx % 2).flatten()
    lj_v = (k_idx % 2).flatten()
    block_v = rb_v * 2 + cb_v
    local_v = li_v * 2 + lj_v
    flat_v = j_flat_v * p + k_flat_v
    
    i_idx_w, k_idx_w = np.indices((n, p))
    i_flat_w = i_idx_w.flatten()
    k_flat_w = k_idx_w.flatten()
    rb_w = (i_idx_w // 2).flatten()
    cb_w = (k_idx_w // 2).flatten()
    li_w = (i_idx_w % 2).flatten()
    lj_w = (k_idx_w % 2).flatten()
    block_w = rb_w * 2 + cb_w
    local_w = li_w * 2 + lj_w
    flat_w = k_flat_w * n + i_flat_w
    
    for K in range(7):
        for l in range(7):
            r = K * 7 + l
            
            # U
            coeff_u = S_A[K, block_u] * S_A[l, local_u]
            U[flat_u, r] = coeff_u
            
            # V
            coeff_v = S_B[K, block_v] * S_B[l, local_v]
            V[flat_v, r] = coeff_v
            
            # W
            coeff_w = S_C[K, block_w] * S_C[l, local_w]
            W[flat_w, r] = coeff_w
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
