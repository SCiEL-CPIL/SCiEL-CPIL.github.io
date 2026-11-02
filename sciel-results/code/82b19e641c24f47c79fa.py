import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (16, rank)
    
    # Rank is 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Strassen steps for 2x2x2
    # Each step is a tuple of (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (local_i, local_j, val)
    # V_coeffs: list of (local_k, local_l, val)
    # W_coeffs: list of (local_i, local_l, val)
    
    steps = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
    ]
    
    for u in range(2): # A row-block index (0, 1)
        for v in range(4): # B col-block index (0, 1, 2, 3)
            for step in steps:
                u_coeffs, v_coeffs, w_coeffs = step
                
                # Fill U
                for li, lj, val in u_coeffs:
                    # Global A index: row i, col 2u + lj
                    gi = li
                    gj = 2 * u + lj
                    u_idx = gi * m + gj
                    U[u_idx][col_idx] += val
                
                # Fill V
                for lk, ll, val in v_coeffs:
                    # Global B index: row 2u + lk, col 2v + ll
                    # Note: B is 4x8. u determines row block (2 rows).
                    # v determines col block (2 cols).
                    gr = 2 * u + lk
                    gc = 2 * v + ll
                    v_idx = gr * p + gc
                    V[v_idx][col_idx] += val
                
                # Fill W
                for li, ll, val in w_coeffs:
                    # Global C index: row li, col 2v + ll
                    # W index: col * n + row
                    gi = li
                    gc = 2 * v + ll
                    w_idx = gc * n + gi
                    W[w_idx][col_idx] += val
                
                col_idx += 1
    
    return {"U": U, "V": V, "W": W}
