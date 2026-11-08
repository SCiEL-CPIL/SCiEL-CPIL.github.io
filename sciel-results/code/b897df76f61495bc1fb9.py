DIMENSIONS = [2, 4, 8]

import time
_START_TIME = time.time()
_DEADLINE = 120

# EVOLVE-BLOCK-START
def construct():
    global _START_TIME, _DEADLINE
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition patterns
    # Local indices: A: (i,j) -> idx = i*2+j, B: (j,k) -> idx = j*2+k, C: (k,i) -> idx = k*2+i
    strassen_U = [
        [(0, 1), (3, 1)],    # r=0: (A11+A22)
        [(2, 1), (3, 1)],    # r=1: (A21+A22)
        [(0, 1)],            # r=2: A11
        [(3, 1)],            # r=3: A22
        [(0, 1), (1, 1)],    # r=4: (A11+A12)
        [(2, 1), (0, -1)],   # r=5: (A21-A11)
        [(1, 1), (3, -1)]    # r=6: (A12-A22)
    ]
    strassen_V = [
        [(0, 1), (3, 1)],    # r=0: (B11+B22)
        [(0, 1)],            # r=1: B11
        [(1, 1), (3, -1)],   # r=2: (B21-B22)
        [(2, 1), (0, -1)],   # r=3: (B12-B11)
        [(3, 1)],            # r=4: B22
        [(0, 1), (1, 1)],    # r=5: (B11+B12)
        [(2, 1), (3, 1)]     # r=6: (B21+B22)
    ]
    strassen_W = [
        [(0, 1), (3, 1)],    # r=0: C11+C22
        [(1, 1), (3, -1)],   # r=1: C12-C22
        [(2, 1), (3, 1)],    # r=2: C21+C22
        [(0, 1), (1, 1)],    # r=3: C11+C12
        [(0, -1), (2, 1)],   # r=4: C21-C11
        [(3, 1)],            # r=5: C22
        [(0, 1)]             # r=6: C11
    ]
    
    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            cr = base_rank + r
            for idx, val in strassen_U[r]:
                li, lj = divmod(idx, 2)
                gi, gj = li + i_offset, lj + j_offset
                U[gi * m + gj][cr] += val
            for idx, val in strassen_V[r]:
                lj, lk = divmod(idx, 2)
                gj, gk = lj + j_offset, lk + k_offset
                V[gj * p + gk][cr] += val
            for idx, val in strassen_W[r]:
                lk, li = divmod(idx, 2)
                gk, gi = lk + k_offset, li + i_offset
                W[gk * n + gi][cr] += val

    # 2x4 * 4x8: split j into [0,2) and [2,4), k into four 2-blocks
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            base = j_off * 4 + k_off // 2 * 7
            add_block(base, 0, j_off, k_off)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
