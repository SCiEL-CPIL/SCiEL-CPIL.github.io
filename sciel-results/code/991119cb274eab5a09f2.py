import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = 2, 3, 8
    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # r, u, v, w
    # u: list of (i, j, val)
    # v: list of (j, k, val)
    # w: list of (k, i, val)
    strassen_factors = [
        # M1
        {
            "u": [(0, 0, 1), (1, 1, 1)],
            "v": [(0, 0, 1), (1, 1, 1)],
            "w": [(0, 0, 1), (1, 1, 1)] # C00, C11
        },
        # M2
        {
            "u": [(1, 0, 1), (1, 1, 1)],
            "v": [(0, 0, 1)],
            "w": [(0, 1, 1)] # C10
        },
        # M3
        {
            "u": [(0, 0, 1)],
            "v": [(0, 1, 1), (1, 1, -1)],
            "w": [(1, 0, 1)] # C01
        },
        # M4
        {
            "u": [(1, 1, 1)],
            "v": [(1, 0, 1), (0, 0, -1)],
            "w": [(0, 0, 1), (0, 1, 1)] # C00, C10
        },
        # M5
        {
            "u": [(0, 0, 1), (0, 1, 1)],
            "v": [(1, 1, 1)],
            "w": [(0, 0, -1), (1, 0, 1)] # C00, C01
        },
        # M6
        {
            "u": [(1, 0, 1), (0, 0, -1)],
            "v": [(0, 0, 1), (0, 1, 1)],
            "w": [(1, 1, 1)] # C11
        },
        # M7
        {
            "u": [(0, 1, 1), (1, 1, -1)],
            "v": [(1, 0, 1), (1, 1, 1)],
            "w": [(0, 0, 1)] # C00
        }
    ]
    
    # Helper to add a factor to the matrices
    def add_factor(rank_idx, i_offset, j_offset, k_offset, factor):
        for (i, j, val) in factor["u"]:
            U[(i + i_offset) * m + (j + j_offset)][rank_idx] += val
        for (j, k, val) in factor["v"]:
            V[(j + j_offset) * p + (k + k_offset)][rank_idx] += val
        for (k, i, val) in factor["w"]:
            W[(k + k_offset) * n + (i + i_offset)][rank_idx] += val

    for s in range(2):
        base_rank = s * 22
        k_base = s * 4
        
        # Part 1: Strassen for j in {0,1}, k' in {0,1}
        for r_idx in range(7):
            rank_idx = base_rank + r_idx
            add_factor(rank_idx, 0, 0, k_base, strassen_factors[r_idx])
            
        # Part 2: Strassen for j in {0,1}, k' in {2,3}
        for r_idx in range(7):
            rank_idx = base_rank + 7 + r_idx
            add_factor(rank_idx, 0, 0, k_base + 2, strassen_factors[r_idx])
            
        # Part 3: Schoolbook for j=2, k' in {0..3}
        for i in range(2):
            for kp in range(4):
                rank_idx = base_rank + 14 + i * 4 + kp
                U[i * m + 2][rank_idx] = 1
                V[2 * p + k_base + kp][rank_idx] = 1
                W[(k_base + kp) * n + i][rank_idx] = 1

    return {"U": U, "V": V, "W": W}
