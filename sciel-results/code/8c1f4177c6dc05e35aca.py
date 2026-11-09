"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    rank = 56
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Each entry is (u_vec, v_vec, w_vec)
    # u_vec, v_vec, w_vec are of length 4 (for 2x2 matrices)
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    col = 0
    # 4 column blocks of C (width 2 each)
    for c in range(4):
        # C indices for this block
        # C_{0, 2c} -> 4c
        # C_{0, 2c+1} -> 4c+2
        # C_{1, 2c} -> 4c+1
        # C_{1, 2c+1} -> 4c+3
        w_offsets = [4*c, 4*c+2, 4*c+1, 4*c+3]
        
        # Sub-problem 1: A_L (cols 0,1) * B_T (rows 0,1 of B_c)
        # A_L indices: 0,1,2,3
        a_offsets = [0, 1, 2, 3]
        # B_T indices in B_c (2x2):
        # B_{0, 2c} -> 2c
        # B_{0, 2c+1} -> 2c+1
        # B_{1, 2c} -> 8+2c
        # B_{1, 2c+1} -> 8+2c+1
        b_offsets = [2*c, 2*c+1, 8+2*c, 8+2*c+1]
        
        for u, v, w in strassen_terms:
            for idx_a, val in enumerate(u):
                if val != 0:
                    U[a_offsets[idx_a]][col] = val
            for idx_b, val in enumerate(v):
                if val != 0:
                    V[b_offsets[idx_b]][col] = val
            for idx_c, val in enumerate(w):
                if val != 0:
                    W[w_offsets[idx_c]][col] = val
            col += 1
            
        # Sub-problem 2: A_R (cols 2,3) * B_B (rows 2,3 of B_c)
        # A_R indices: 4,5,6,7
        a_offsets = [4, 5, 6, 7]
        # B_B indices in B_c (2x2):
        # B_{2, 2c} -> 16+2c
        # B_{2, 2c+1} -> 16+2c+1
        # B_{3, 2c} -> 24+2c
        # B_{3, 2c+1} -> 24+2c+1
        b_offsets = [16+2*c, 16+2*c+1, 24+2*c, 24+2*c+1]
        
        for u, v, w in strassen_terms:
            for idx_a, val in enumerate(u):
                if val != 0:
                    U[a_offsets[idx_a]][col] = val
            for idx_b, val in enumerate(v):
                if val != 0:
                    V[b_offsets[idx_b]][col] = val
            for idx_c, val in enumerate(w):
                if val != 0:
                    W[w_offsets[idx_c]][col] = val
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
