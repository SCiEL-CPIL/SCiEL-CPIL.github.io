import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # Rank = 42 (6 blocks * 7 Strassen terms)
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block definitions
    # b: 0..5
    # j_off: 0 or 2
    # k_off: 0, 2, 4
    blocks = [
        (0, 0), (0, 2), (0, 4),
        (2, 0), (2, 2), (2, 4)
    ]
    
    # Strassen coefficients
    # s: 1..7
    # U_coeffs: list of (u, v, coeff)
    # V_coeffs: list of (v, w, coeff)
    # W_coeffs: list of (u, w, coeff)
    
    strassen_U = {
        1: [(0, 0, 1), (1, 1, 1)],
        2: [(1, 0, 1), (1, 1, 1)],
        3: [(0, 0, 1)],
        4: [(1, 1, 1)],
        5: [(0, 0, 1), (0, 1, 1)],
        6: [(1, 0, 1), (0, 0, -1)],
        7: [(0, 1, 1), (1, 1, -1)]
    }
    
    strassen_V = {
        1: [(0, 0, 1), (1, 1, 1)],
        2: [(0, 0, 1)],
        3: [(0, 1, 1), (1, 1, -1)],
        4: [(1, 0, 1), (0, 0, -1)],
        5: [(1, 1, 1)],
        6: [(0, 0, 1), (0, 1, 1)],
        7: [(1, 0, 1), (1, 1, 1)]
    }
    
    strassen_W = {
        1: [(0, 0, 1), (1, 1, 1)],
        2: [(1, 0, 1), (1, 1, -1)],
        3: [(0, 1, 1), (1, 1, 1)],
        4: [(0, 0, 1), (1, 0, 1)],
        5: [(0, 0, -1), (0, 1, 1)],
        6: [(1, 1, 1)],
        7: [(0, 0, 1)]
    }
    
    for b_idx, (j_off, k_off) in enumerate(blocks):
        for s in range(1, 8):
            c = b_idx * 7 + (s - 1)
            
            # Update U
            for u, v, coeff in strassen_U[s]:
                row_u = 4 * u + (j_off + v)
                U[row_u][c] += coeff
            
            # Update V
            for v, w, coeff in strassen_V[s]:
                row_v = 6 * (j_off + v) + (k_off + w)
                V[row_v][c] += coeff
            
            # Update W
            for u, w, coeff in strassen_W[s]:
                row_w = 2 * (k_off + w) + u
                W[row_w][c] += coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
