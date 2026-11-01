"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_map, v_map, w_map):
        nonlocal col
        for u_idx, val in u_map:
            U[u_idx][col] += val
        for v_idx, val in v_map:
            V[v_idx][col] += val
        for w_idx, val in w_map:
            W[w_idx][col] += val
        col += 1

    # Strassen terms for 2x2x2
    # Local indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    def add_strassen(u_offset, v_offsets, w_offsets):
        # u_offset: base index for U
        # v_offsets: list of 4 offsets for V
        # w_offsets: list of 4 offsets for W
        
        # m1
        add_term([(u_offset+0, 1), (u_offset+3, 1)], 
                 [(v_offsets[0], 1), (v_offsets[3], 1)], 
                 [(w_offsets[0], 1), (w_offsets[3], 1)])
        # m2
        add_term([(u_offset+2, 1), (u_offset+3, 1)], 
                 [(v_offsets[0], 1)], 
                 [(w_offsets[2], 1)])
        # m3
        add_term([(u_offset+0, 1)], 
                 [(v_offsets[1], 1), (v_offsets[3], -1)], 
                 [(w_offsets[1], 1)])
        # m4
        add_term([(u_offset+3, 1)], 
                 [(v_offsets[2], 1), (v_offsets[0], -1)], 
                 [(w_offsets[0], 1)])
        # m5
        add_term([(u_offset+0, 1), (u_offset+1, 1)], 
                 [(v_offsets[3], 1)], 
                 [(w_offsets[0], -1)])
        # m6
        add_term([(u_offset+2, 1), (u_offset+0, -1)], 
                 [(v_offsets[0], 1), (v_offsets[1], 1)], 
                 [(w_offsets[3], 1)])
        # m7
        add_term([(u_offset+1, 1), (u_offset+2, -1)], 
                 [(v_offsets[2], 1), (v_offsets[3], 1)], 
                 [(w_offsets[0], 1)])

    # Trivial terms for 2x2x1
    def add_trivial(u_offset, v_offsets, w_offsets):
        # v_offsets: list of 2 offsets for V
        # w_offsets: list of 2 offsets for W
        add_term([(u_offset+0, 1)], [(v_offsets[0], 1)], [(w_offsets[0], 1)])
        add_term([(u_offset+1, 1)], [(v_offsets[1], 1)], [(w_offsets[0], 1)])
        add_term([(u_offset+2, 1)], [(v_offsets[0], 1)], [(w_offsets[1], 1)])
        add_term([(u_offset+3, 1)], [(v_offsets[1], 1)], [(w_offsets[1], 1)])

    # Term 1: A_L * B_T
    # A_L is rows 0,1 cols 0,1 of A. U indices 0,1,2,3.
    u1 = 0
    # B_T is rows 0,1 of B. V indices:
    # Y_1 (cols 0,1): V[0], V[1], V[5], V[6]
    # Y_2 (cols 2,3): V[2], V[3], V[7], V[8]
    # Y_3 (col 4): V[4], V[9]
    v1_y1 = [0, 1, 5, 6]
    v1_y2 = [2, 3, 7, 8]
    v1_y3 = [4, 9]
    # C is 2x5. W indices:
    # Z_1 (cols 0,1): W[0], W[1], W[5], W[6]
    # Z_2 (cols 2,3): W[2], W[3], W[7], W[8]
    # Z_3 (col 4): W[4], W[9]
    w1_z1 = [0, 1, 5, 6]
    w1_z2 = [2, 3, 7, 8]
    w1_z3 = [4, 9]
    
    add_strassen(u1, v1_y1, w1_z1)
    add_strassen(u1, v1_y2, w1_z2)
    add_trivial(u1, v1_y3, w1_z3)

    # Term 2: A_R * B_B
    # A_R is rows 0,1 cols 2,3 of A. U indices 4,5,6,7.
    u2 = 4
    # B_B is rows 2,3 of B. V indices:
    # Y_1 (cols 0,1): V[10], V[11], V[15], V[16]
    # Y_2 (cols 2,3): V[12], V[13], V[17], V[18]
    # Y_3 (col 4): V[14], V[19]
    v2_y1 = [10, 11, 15, 16]
    v2_y2 = [12, 13, 17, 18]
    v2_y3 = [14, 19]
    # C is 2x5. W indices:
    # Z_1 (cols 0,1): W[0], W[1], W[5], W[6]
    # Z_2 (cols 2,3): W[2], W[3], W[7], W[8]
    # Z_3 (col 4): W[4], W[9]
    w2_z1 = [0, 1, 5, 6]
    w2_z2 = [2, 3, 7, 8]
    w2_z3 = [4, 9]
    
    add_strassen(u2, v2_y1, w2_z1)
    add_strassen(u2, v2_y2, w2_z2)
    add_trivial(u2, v2_y3, w2_z3)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
