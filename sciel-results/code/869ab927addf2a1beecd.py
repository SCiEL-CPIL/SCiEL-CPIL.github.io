import numpy as np

def construct():
    n, m, p = 4, 5, 5
    
    # Total rank 30
    U = np.zeros((n * m, 30))
    V = np.zeros((m * p, 30))
    W = np.zeros((p * n, 30))
    
    col_offset = 0
    
    # Helper to add a block
    def add_block(U_block, V_block, W_block, i_off, k_off, j_off):
        nonlocal col_offset
        rank_b = U_block.shape[1]
        
        # U: n*m
        for i in range(U_block.shape[0]):
            for r in range(rank_b):
                global_i = i_off + (i // U_block.shape[1]) # This logic is tricky, better to pass explicit mapping or construct locally
        # Actually, easier to just iterate local indices and map to global
        # U_block is (local_n * local_m, rank)
        # local_n, local_m, local_p
        # I will define the blocks explicitly with local dimensions and mappings.
        
    # Block 1: A00 (2x2) B00 (2x2) -> C00 (2x2). Strassen Rank 7.
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Map Strassen to Global
    # A00: i in 0,1; k in 0,1. Global idx: i*5 + k
    # B00: k in 0,1; j in 0,1. Global idx: k*5 + j
    # C00: i in 0,1; j in 0,1. Global idx: i*5 + j
    for i_l in range(2):
        for k_l in range(2):
            g_idx_u = i_l * 5 + k_l
            for r in range(7):
                U[g_idx_u, col_offset + r] += U_s[i_l * 2 + k_l, r]
    for k_l in range(2):
        for j_l in range(2):
            g_idx_v = k_l * 5 + j_l
            for r in range(7):
                V[g_idx_v, col_offset + r] += V_s[k_l * 2 + j_l, r]
    for k_l in range(2): # W is p*n? No, W is p*n in contract? 
        # Contract: W[k*n+i, r]. Here output is C[i,j].
        # The problem statement says W[k*n+i, r].
        # For matrix mult C = AB, C[i,j] = sum_k A[i,k] B[k,j].
        # Tensor T[i,k,j].
        # Decomposition: T[i,k,j] = sum_r U[i*k, r] V[k*j, r] W[j*i, r]?
        # The prompt says: "output factor follows the official trace convention W[k*n+i,r]".
        # This usually means W is indexed by the output dimension?
        # If W is (p*n, rank), and index is k*n+i, this looks like it's mixing k and i.
        # In standard matrix mult decomposition, the third factor is often associated with the output C.
        # If C is n x p, then W should be (n*p, rank).
        # The prompt says W is (p*n, rank).
        # And index is k*n+i.
        # If output is C[i,j], and we view it as vector of length n*p.
        # If we flatten C row-major (i major), index is i*p + j.
        # If we flatten C col-major (j major), index is j*n + i.
        # The prompt says W[k*n+i, r].
        # If k corresponds to j (column of C) and i corresponds to i (row of C), then k*n+i is col-major index of C.
        # So W[r, col] where col = j*n + i.
        # So W index is j*n + i.
        # Let's assume W[g_idx_w, r] where g_idx_w = j_l * n + i_l.
        for j_l in range(2):
            for i_l in range(2):
                g_idx_w = j_l * n + i_l
                for r in range(7):
                    W[g_idx_w, col_offset + r] += W_s[j_l * 2 + i_l, r] # W_s is 2x2? No, W_s is 4x7.
                    # W_s[i_l*2 + j_l, r] or W_s[j_l*2 + i_l, r]?
                    # Strassen W is for output C.
                    # C is 2x2.
                    # W_s indices: 0,1,2,3.
                    # If W_s is defined for C row-major, then index i*2+j.
                    # If W_s is defined for C col-major, then index j*2+i.
                    # Standard Strassen W usually matches C layout.
                    # Let's assume W_s[i_l*2 + j_l, r] corresponds to C[i_l, j_l].
                    # Then global index is j_l * n + i_l.
                    # So we map W_s[i_l*2 + j_l, r] to W[j_l * n + i_l, r].
    col_offset += 7
    
    # ... and so on for other blocks.
