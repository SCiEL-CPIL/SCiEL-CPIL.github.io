"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen rank-7 coefficients for 2x2x2: (U_part, V_part, W_part)
    # Each part is a list of (index, coeff) for the respective matrix
    S = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)],           [(0,1,1),(1,1,-1)]),
        ([(0,0,1)],           [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        ([(1,1,1)],           [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        ([(0,0,1),(0,1,1)],   [(1,1,1)],           [(0,0,-1),(1,0,1)]),
        ([(1,0,1),(0,0,-1)],  [(0,0,1),(0,1,1)],   [(1,1,1)]),
        ([(0,1,1),(1,1,-1)],  [(1,0,1),(1,1,1)],   [(0,0,1)]),
    ]
    
    idx = 0
    for out_blk in range(4):
        for mid_blk in range(2):
            a_off = mid_blk * 2
            b_row_off = mid_blk * 2
            b_col_off = out_blk * 2
            
            for u_terms, v_terms, w_terms in S:
                for (r, c, w) in u_terms:
                    U[r * m + a_off + c][idx] += w
                for (r, c, w) in v_terms:
                    V[(b_row_off + r) * p + b_col_off + c][idx] += w
                for (r, c, w) in w_terms:
                    W[(b_col_off + c) * n + r][idx] += w
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
