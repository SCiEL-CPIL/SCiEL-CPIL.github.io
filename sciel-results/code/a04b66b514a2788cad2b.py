"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 blocks * 7 Strassen terms = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Block offsets for A (2x4), B (4x4), C (2x4 -> W is 4x2)
    # A: i in {0,1}, j in {0..3}
    # B: j in {0..3}, k in {0..3}
    # C: i in {0,1}, k in {0..3} -> W index k*n + i
    
    blocks = [
        # (A_j_offset, B_j_offset, B_k_offset, C_k_offset)
        (0, 0, 0, 0), # Block 0: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
        (2, 2, 0, 0), # Block 1: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
        (0, 0, 2, 2), # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
        (2, 2, 2, 2), # Block 3: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    ]
    
    for b_idx, (a_j_off, b_j_off, b_k_off, c_k_off) in enumerate(blocks):
        for s_idx in range(7):
            col = b_idx * 7 + s_idx
            u_coeffs, v_coeffs, w_coeffs = terms[s_idx]
            
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                i = li
                j = a_j_off + lj
                g_idx = i * m + j
                U[g_idx][col] += coeff
                
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                j = b_j_off + li
                k = b_k_off + lj
                g_idx = j * p + k
                V[g_idx][col] += coeff
                
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                i = li
                k = c_k_off + lj
                g_idx = k * n + i
                W[g_idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
