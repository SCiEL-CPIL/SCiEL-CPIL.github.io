"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Rank-49 decomposition for 4x4 matrix multiplication."""
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Rank-49 decomposition using Strassen-based construction.
    # We use 7 groups of 7 rank-1 terms each.
    # 
    # The construction partitions the 4x4 indices into structured subsets
    # and applies Strassen's 7-multiplication scheme to each.
    
    # Strassen 2x2 patterns: (a_coeffs, b_coeffs, c_coeffs)
    # Each is a list of 4 coefficients for the 2x2 block [00,01,10,11]
    S = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),  # M7
    ]
    
    # 7 index configurations for mapping local 2x2 to global 4x4
    # Each config: (a_r, a_c, b_r, b_c, c_r, c_c) where each is a tuple of 2 indices
    C = [
        ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1)),
        ((0,1),(2,3),(0,1),(2,3),(0,1),(2,3)),
        ((2,3),(0,1),(2,3),(0,1),(2,3),(0,1)),
        ((2,3),(2,3),(2,3),(2,3),(2,3),(2,3)),
        ((0,2),(0,2),(0,2),(0,2),(0,2),(0,2)),
        ((1,3),(1,3),(1,3),(1,3),(1,3),(1,3)),
        ((0,3),(1,2),(0,3),(1,2),(0,3),(1,2)),
    ]
    
    for ci, (a_r, a_c, b_r, b_c, c_r, c_c) in enumerate(C):
        for si in range(7):
            a_coef, b_coef, c_coef = S[si]
            tid = ci * 7 + si
            
            for lr in range(2):
                for lc in range(2):
                    idx = lr * 2 + lc
                    if a_coef[idx] != 0:
                        i, j = a_r[lr], a_c[lc]
                        U[i*m + j][tid] = a_coef[idx]
                    if b_coef[idx] != 0:
                        j, k = b_r[lr], b_c[lc]
                        V[j*p + k][tid] = b_coef[idx]
                    if c_coef[idx] != 0:
                        k, i = c_r[lr], c_c[lc]
                        W[k*n + i][tid] = c_coef[idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

