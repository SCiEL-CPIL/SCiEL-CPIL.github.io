import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Use Strassen-based rank 56 decomposition
    # 2x4x8 splits into two 2x2x8
    # 2x2x8 splits into four 2x2x2
    # Total 8 blocks of 2x2x2 Strassen (rank 7)
    # Total rank 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    strassen_U = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)],
    ]
    strassen_V = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
    ]
    strassen_W = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)],
    ]
    
    def add_strassen_block(base_rank, i_off, j_off, k_off):
        for r in range(7):
            cr = base_rank + r
            for idx, val in strassen_U[r]:
                li, lj = divmod(idx, 2)
                U[(li + i_off) * 4 + (lj + j_off)][cr] += val
            for idx, val in strassen_V[r]:
                lj, lk = divmod(idx, 2)
                V[(lj + j_off) * 8 + (lk + k_off)][cr] += val
            for idx, val in strassen_W[r]:
                lk, li = divmod(idx, 2)
                W[(lk + k_off) * 2 + (li + i_off)][cr] += val
    
    # 8 Strassen blocks for 2x4x8
    # j_block: 0, 1 (splits m=4 into two 2s)
    # k_block: 0, 1, 2, 3 (splits p=8 into four 2s)
    for j_block in range(2):
        for k_block in range(4):
            base = (j_block * 4 + k_block) * 7
            add_strassen_block(base, 0, j_block * 2, k_block * 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
