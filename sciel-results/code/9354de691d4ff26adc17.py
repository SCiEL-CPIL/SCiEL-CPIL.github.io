import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = _t0 + 145.0

    n, m, p = 6, 6, 6

    # Use Strassen 2x2 tiling (Rank 189)
    block_size = 2
    blocks_per_dim = 3
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    offsets = [i * block_size for i in range(blocks_per_dim)]
    
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        off_I = offsets[I]
        for J in range(blocks_per_dim):
            off_J = offsets[J]
            for K in range(blocks_per_dim):
                off_K = offsets[K]
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = off_I + pi
                        c = off_J + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = off_J + pi
                        c = off_K + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = off_K + pj
                        i = off_I + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
                if time.time() > _deadline:
                    break
            if time.time() > _deadline:
                break
        if time.time() > _deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
