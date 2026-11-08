import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Rank 56 decomposition using 8 Strassen blocks of 2x2x2
    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2 coefficients
    # U, V, W entries for the 7 products
    # Indices in 2x2 matrix: 0:00, 1:01, 2:10, 3:11
    S = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    def add_strassen_block(base_rank, i_off, j_off, k_off):
        for r in range(7):
            cr = base_rank + r
            u_entries, v_entries, w_entries = S[r]
            for idx, val in u_entries:
                li, lj = divmod(idx, 2)
                gi, gj = li + i_off, lj + j_off
                U[gi * m + gj, cr] += val
            for idx, val in v_entries:
                lj, lk = divmod(idx, 2)
                gj, gk = lj + j_off, lk + k_off
                V[gj * p + gk, cr] += val
            for idx, val in w_entries:
                lk, li = divmod(idx, 2)
                gk, gi = lk + k_off, li + i_off
                W[gk * n + gi, cr] += val

    # 8 blocks: 1 (i) x 2 (j) x 4 (k)
    block_idx = 0
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            add_strassen_block(block_idx * 7, 0, j_off, k_off)
            block_idx += 1
            
    return {"U": U, "V": V, "W": W}
