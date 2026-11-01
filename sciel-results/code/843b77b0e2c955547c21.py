"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for a 2x2 block
    # U rows: 0:a, 1:b, 2:c, 3:d
    # V rows for block b (cols 2b, 2b+1 of B):
    #   b=0: u=e(0), v=f(1), w=k(6), x=l(7)
    #   b=1: u=g(2), v=h(3), w=m(8), x=n(9)
    #   b=2: u=i(4), v=j(5), w=o(10), x=p(11)
    # W rows for block b (cols 2b, 2b+1 of C):
    #   C_{0, 2b} is row 2b
    #   C_{1, 2b} is row 2b+1
    #   C_{0, 2b+1} is row 2b+2
    #   C_{1, 2b+1} is row 2b+3
    
    # Term definitions: (U_coeffs, V_offsets, W_offsets)
    # V_offsets: (idx_u, val_u, idx_v, val_v, ...) relative to block start in V?
    # It's easier to hardcode the indices for each block.
    
    def set_term(col, u_vec, v_pairs, w_pairs):
        for i, val in enumerate(u_vec):
            if val != 0:
                U[i][col] = val
        for idx, val in v_pairs:
            V[idx][col] = val
        for idx, val in w_pairs:
            W[idx][col] = val

    for b in range(3):
        # V indices for this block
        # u is at row 2*b
        # v is at row 2*b + 1
        # w is at row 6 + 2*b
        # x is at row 7 + 2*b
        idx_u = 2 * b
        idx_v = 2 * b + 1
        idx_w = 6 + 2 * b
        idx_x = 7 + 2 * b
        
        # W indices for this block
        # C_{0, 2b} is row 2*b
        # C_{1, 2b} is row 2*b + 1
        # C_{0, 2b+1} is row 2*b + 2
        # C_{1, 2b+1} is row 2*b + 3
        idx_c00 = 2 * b
        idx_c10 = 2 * b + 1
        idx_c01 = 2 * b + 2
        idx_c11 = 2 * b + 3
        
        # Term 1: (a+d)(u+x)
        col = b * 7 + 0
        set_term(col, [1, 0, 0, 1], 
                 [(idx_u, 1), (idx_x, 1)], 
                 [(idx_c00, 1), (idx_c11, 1)])
                 
        # Term 2: (c+d)u
        col = b * 7 + 1
        set_term(col, [0, 0, 1, 1], 
                 [(idx_u, 1)], 
                 [(idx_c10, 1), (idx_c01, 1)])
                 
        # Term 3: a(w-x)
        col = b * 7 + 2
        set_term(col, [1, 0, 0, 0], 
                 [(idx_w, 1), (idx_x, -1)], 
                 [(idx_c01, 1), (idx_c11, 1)])
                 
        # Term 4: d(v-w)
        col = b * 7 + 3
        set_term(col, [0, 0, 0, 1], 
                 [(idx_v, 1), (idx_w, -1)], 
                 [(idx_c00, 1)])
                 
        # Term 5: (a+b)x
        col = b * 7 + 4
        set_term(col, [1, 1, 0, 0], 
                 [(idx_x, 1)], 
                 [(idx_c00, -1), (idx_c10, 1)])
                 
        # Term 6: (c-d)(v+x)
        col = b * 7 + 5
        set_term(col, [0, 0, 1, -1], 
                 [(idx_v, 1), (idx_x, 1)], 
                 [(idx_c11, 1)])
                 
        # Term 7: (b-c)(u+w)
        col = b * 7 + 6
        set_term(col, [0, 1, -1, 0], 
                 [(idx_u, 1), (idx_w, 1)], 
                 [(idx_c00, 1)])
                 
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
