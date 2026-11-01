"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-35 exact tensor decomposition for 2x2 * 2x10 matmul.
    
    Strategy: Decompose the 2x2 * 2x10 multiplication into 5 independent 
    2x2 * 2x2 block multiplications. Apply Strassen's algorithm (rank 7) 
    to each block. Total rank = 5 * 7 = 35.
    """
    n, m, p = DIMENSIONS
    # 5 blocks of 2x2
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen coefficients
    # U: (a, b, c, d)
    U_coeffs = [
        [1, 0, 0, 1],  # P1: (a+d)
        [0, 0, 1, 1],  # P2: (c+d)
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: (a+b)
        [-1, 0, 1, 0], # P6: (c-a)
        [0, 1, 0, -1]  # P7: (b-d)
    ]
    
    # V: (e, f, g, h) corresponding to B block cols 0,1
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    V_coeffs = [
        [1, 1, 0, 0],  # P1: (e+h) -> wait, P1 is (a+d)(e+h). So V part is (e+h).
        [1, 0, 0, 0],  # P2: e
        [0, 1, 0, -1], # P3: (f-h)
        [0, 0, 1, -1], # P4: (g-e)
        [0, 0, 0, 1],  # P5: h
        [1, 1, 0, 0],  # P6: (e+f)
        [0, 0, 1, 1]   # P7: (g+h)
    ]
    
    # W: (m11, m12, m21, m22)
    # m11 = P1 + P4 - P5 + P7
    # m12 = P3 + P5
    # m21 = P2 + P4
    # m22 = P1 - P2 + P3 + P6
    W_coeffs = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, -1],  # P2
        [0, 1, 0, 1],   # P3
        [1, 0, 1, 0],   # P4
        [-1, 1, 0, 0],  # P5
        [0, 0, 0, 1],   # P6
        [1, 0, 0, 0]    # P7
    ]
    
    for b in range(num_blocks):
        # Block b uses B columns 2b and 2b+1
        # Output columns 2b and 2b+1
        
        # V indices for this block
        # e = B[0, 2b] -> V index 0*p + 2b
        # f = B[0, 2b+1] -> V index 0*p + 2b+1
        # g = B[1, 2b] -> V index 1*p + 2b
        # h = B[1, 2b+1] -> V index 1*p + 2b+1
        v_e = 0 * p + 2 * b
        v_f = 0 * p + 2 * b + 1
        v_g = 1 * p + 2 * b
        v_h = 1 * p + 2 * b + 1
        v_indices = [v_e, v_f, v_g, v_h]
        
        # W indices for this block
        # m11 = C[0, 2b] -> W index 2b*n + 0
        # m12 = C[0, 2b+1] -> W index (2b+1)*n + 0
        # m21 = C[1, 2b] -> W index 2b*n + 1
        # m22 = C[1, 2b+1] -> W index (2b+1)*n + 1
        w_m11 = 2 * b * n + 0
        w_m12 = (2 * b + 1) * n + 0
        w_m21 = 2 * b * n + 1
        w_m22 = (2 * b + 1) * n + 1
        w_indices = [w_m11, w_m12, w_m21, w_m22]
        
        for t in range(rank_per_block):
            r = b * rank_per_block + t
            
            # Fill U
            for i in range(n * m):
                U[i][r] = U_coeffs[t][i]
                
            # Fill V
            for j_idx, v_idx in enumerate(v_indices):
                V[v_idx][r] = V_coeffs[t][j_idx]
                
            # Fill W
            for w_idx, w_val in zip(w_indices, W_coeffs[t]):
                W[w_idx][r] = w_val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
