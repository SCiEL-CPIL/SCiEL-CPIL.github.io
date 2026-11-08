DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_U = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    strassen_V = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    strassen_W = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)]
    ]
    
    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val

    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            base = (j_off // 2) * 28 + (k_off // 2) * 7
            add_block(base, 0, j_off, k_off)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
