"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    block_size = 2
    n_blocks = n // block_size  # 3

    # Rank-189 block Strassen: 3x3 blocks of 2x2, each 2x2 uses rank-7 Strassen
    rank = n_blocks ** 3 * 7

    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    col = 0

    # Index helpers
    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    # Strassen 2x2 multiplication terms:
    # M1 = (A00+A11)(B00+B11) -> C00+C11
    # M2 = (A10+A11)(B00)     -> C10+C11
    # M3 = (A00)(B01-B11)     -> C01+C11
    # M4 = (A11)(B10-B00)     -> C00+C10
    # M5 = (A00+A01)(B11)     -> C00-C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00

    # Each term: list of (u_entries, v_entries, w_entries)
    # u_entries: [(r, c, coeff), ...] for A
    # v_entries: [(r, c, coeff), ...] for B
    # w_entries: [(r, c, coeff), ...] for C
    strassen_terms = [
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                for u_entries, v_entries, w_entries in strassen_terms:
                    for (r, c, coeff) in u_entries:
                        U[u_idx(ib, kb, r, c), col] += coeff
                    for (r, c, coeff) in v_entries:
                        V[v_idx(kb, jb, r, c), col] += coeff
                    for (r, c, coeff) in w_entries:
                        W[w_idx(ib, jb, r, c), col] += coeff
                    col += 1

    if time.time() > deadline:
        # Should not happen, but safety
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
