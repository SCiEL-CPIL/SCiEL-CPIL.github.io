import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * (2 * 2 * 4) = 112
    rank = 112
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients
    # s: 0 to 6
    # L_coeffs[s][u][v] is list of (bi, bj, coeff) for A blocks
    # R_coeffs[s][v][w] is list of (bi, bj, coeff) for B blocks
    # C_coeffs[s] is list of (ci, cj, coeff) for C blocks
    
    L_coeffs = [None] * 7
    R_coeffs = [None] * 7
    C_coeffs = [None] * 7
    
    # s=0: M1
    L_coeffs[0] = [
        [[(0,0,1), (1,1,1)] for _ in range(2)], # u=0
        [[(0,0,1), (1,1,1)] for _ in range(2)]  # u=1
    ]
    R_coeffs[0] = [
        [[(0,0,1), (1,1,1)] for _ in range(4)], # v=0
        [[(0,0,1), (1,1,1)] for _ in range(4)]  # v=1
    ]
    C_coeffs[0] = [(0,0,1), (1,1,1)]
    
    # s=1: M2
    L_coeffs[1] = [
        [[(1,0,1), (1,1,1)] for _ in range(2)],
        [[(1,0,1), (1,1,1)] for _ in range(2)]
    ]
    R_coeffs[1] = [
        [[(0,0,1)] for _ in range(4)],
        [[(0,0,1)] for _ in range(4)]
    ]
    C_coeffs[1] = [(1,0,1), (1,1,-1)]
    
    # s=2: M3
    L_coeffs[2] = [
        [[(0,0,1)] for _ in range(2)],
        [[(0,0,1)] for _ in range(2)]
    ]
    R_coeffs[2] = [
        [[(0,1,1), (1,1,-1)] for _ in range(4)],
        [[(0,1,1), (1,1,-1)] for _ in range(4)]
    ]
    C_coeffs[2] = [(0,1,1), (1,1,1)]
    
    # s=3: M4
    L_coeffs[3] = [
        [[(1,1,1)] for _ in range(2)],
        [[(1,1,1)] for _ in range(2)]
    ]
    R_coeffs[3] = [
        [[(1,0,1), (0,0,-1)] for _ in range(4)],
        [[(1,0,1), (0,0,-1)] for _ in range(4)]
    ]
    C_coeffs[3] = [(0,0,1), (1,0,1)]
    
    # s=4: M5
    L_coeffs[4] = [
        [[(0,0,1), (0,1,1)] for _ in range(2)],
        [[(0,0,1), (0,1,1)] for _ in range(2)]
    ]
    R_coeffs[4] = [
        [[(1,1,1)] for _ in range(4)],
        [[(1,1,1)] for _ in range(4)]
    ]
    C_coeffs[4] = [(0,0,-1), (0,1,1)]
    
    # s=5: M6
    L_coeffs[5] = [
        [[(1,0,1), (0,0,-1)] for _ in range(2)],
        [[(1,0,1), (0,0,-1)] for _ in range(2)]
    ]
    R_coeffs[5] = [
        [[(0,0,1), (0,1,1)] for _ in range(4)],
        [[(0,0,1), (0,1,1)] for _ in range(4)]
    ]
    C_coeffs[5] = [(1,1,1)]
    
    # s=6: M7
    L_coeffs[6] = [
        [[(0,1,1), (1,1,-1)] for _ in range(2)],
        [[(0,1,1), (1,1,-1)] for _ in range(2)]
    ]
    R_coeffs[6] = [
        [[(1,0,1), (1,1,1)] for _ in range(4)],
        [[(1,0,1), (1,1,1)] for _ in range(4)]
    ]
    C_coeffs[6] = [(0,0,1)]
    
    col = 0
    for s in range(7):
        for u in range(2):
            for v in range(2):
                for w in range(4):
                    # U vector
                    for bi, bj, coeff in L_coeffs[s][u][v]:
                        # A block (bi, bj) entry (u, v)
                        # Global A row: 2*bi + u, col: 2*bj + v
                        row_a = 2 * bi + u
                        col_a = 2 * bj + v
                        U[row_a * m + col_a][col] += coeff
                    
                    # V vector
                    for bi, bj, coeff in R_coeffs[s][v][w]:
                        # B block (bi, bj) entry (v, w)
                        # Global B row: 2*bi + v, col: 4*bj + w
                        row_b = 2 * bi + v
                        col_b = 4 * bj + w
                        V[row_b * p + col_b][col] += coeff
                    
                    # W vector
                    for ci, cj, coeff in C_coeffs[s]:
                        # C block (ci, cj) entry (u, w)
                        # Global C row: 2*ci + u, col: 4*cj + w
                        row_c = 2 * ci + u
                        col_c = 4 * cj + w
                        # W index: col_c * n + row_c
                        W[col_c * n + row_c][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
