"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 14 (2 blocks of 7 Strassen terms)
    rank = 14
    
    # Shapes
    # U: (n*m, rank) = (4, 14)
    # V: (m*p, rank) = (8, 14)
    # W: (p*n, rank) = (8, 14)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # u: [a00, a01, a10, a11]
    # v: [b00, b01, b10, b11]
    # w: [c00, c01, c10, c11] (local)
    
    strassen_terms = [
        # (u, v, w)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [1, -1, 0, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    # Mapping for Block 1 (cols 0,1)
    # B local [0,1,4,5] -> global [0,1,4,5]
    # C local [0,2,1,3] -> global [0,2,1,3] (c00->0, c01->2, c10->1, c11->3)
    
    # Mapping for Block 2 (cols 2,3)
    # B local [2,3,6,7] -> global [2,3,6,7]
    # C local [4,6,5,7] -> global [4,6,5,7] (c00->4, c01->6, c10->5, c11->7)
    
    col = 0
    for block in range(2):
        # Determine global indices for V and W
        if block == 0:
            v_map = [0, 1, 4, 5]
            w_map = [0, 2, 1, 3]
        else:
            v_map = [2, 3, 6, 7]
            w_map = [4, 6, 5, 7]
            
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            # U is same for both blocks (uses same A)
            for idx, val in enumerate(u_coeffs):
                U[idx][col] = val
                
            # V
            for local_idx, val in enumerate(v_coeffs):
                global_idx = v_map[local_idx]
                V[global_idx][col] = val
                
            # W
            for local_idx, val in enumerate(w_coeffs):
                global_idx = w_map[local_idx]
                W[global_idx][col] = val
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
