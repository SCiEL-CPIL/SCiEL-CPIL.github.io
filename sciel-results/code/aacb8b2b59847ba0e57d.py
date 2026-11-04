import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    strassen_U = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1}
    ]
    
    def add_block(base_rank, i_base, j_base, k_base):
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * 4 + gj][cr] += val
            
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * 8 + gk][cr] += val
            
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * 2 + gi][cr] += val

    # 8 blocks: 2 (i) x 4 (k-blocks of size 2) ? 
    # No, the blocks are defined by (i_base, j_base, k_base) in the 2x2x2 sense.
    # A is 2x4. B is 4x8.
    # We split A into two 2x2 blocks vertically? No, A is 2x4.
    # A = [A0 | A1] where A0, A1 are 2x2.
    # B = [B0; B1] where B0, B1 are 2x8? No.
    # B is 4x8.
    # B = [B00 B01 B02 B03; B10 B11 B12 B13] ?
    # Let's stick to the previous successful mapping:
    # Block (i_b, j_b, k_b) corresponds to A[i_b:i_b+2, j_b:j_b+2] * B[j_b:j_b+2, k_b:k_b+2].
    # A has size 2x4. i_b can be 0. j_b can be 0 or 2.
    # B has size 4x8. j_b can be 0 or 2. k_b can be 0, 2, 4, 6.
    # So we have 1 * 2 * 4 = 8 blocks.
    
    # i_base is always 0.
    # j_base is 0 or 2.
    # k_base is 0, 2, 4, 6.
    
    # Block 0: i=0, j=0, k=0 (A0 * B00)
    add_block(0, 0, 0, 0)
    # Block 1: i=0, j=0, k=2 (A0 * B01)
    add_block(7, 0, 0, 2)
    # Block 2: i=0, j=0, k=4 (A0 * B02)
    add_block(14, 0, 0, 4)
    # Block 3: i=0, j=0, k=6 (A0 * B03)
    add_block(21, 0, 0, 6)
    
    # Block 4: i=0, j=2, k=0 (A1 * B10)
    add_block(28, 0, 2, 0)
    # Block 5: i=0, j=2, k=2 (A1 * B11)
    add_block(35, 0, 2, 2)
    # Block 6: i=0, j=2, k=4 (A1 * B12)
    add_block(42, 0, 2, 4)
    # Block 7: i=0, j=2, k=6 (A1 * B13)
    add_block(49, 0, 2, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
