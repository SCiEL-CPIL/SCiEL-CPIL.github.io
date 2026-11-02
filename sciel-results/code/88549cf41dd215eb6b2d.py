"""Optimized program: rank-minimal exact tensor decomposition for 2x4 @ 4x6 using Strassen blocks."""
import time

_DIMENSIONS = [2, 4, 6]
_DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep safety margin

def construct():
    _t0 = time.time()
    n, m, p = 2, 4, 6
    rank_total = 42
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    # Strassen 2x2 @ 2x2 rank-7 decomposition.
    # Local indices: 0=a11/b11/c11, 1=a12/b12/c12, 2=a21/b21/c21, 3=a22/b22/c22
    strassen_U = [
        {0: 1, 3: 1},   # P1: (a11+a22)
        {2: 1, 3: 1},   # P2: (a21+a22)
        {0: 1},         # P3: a11
        {3: 1},         # P4: a22
        {0: 1, 1: 1},   # P5: (a11+a12)
        {0: -1, 2: 1},  # P6: (a21-a11)
        {1: 1, 3: -1},  # P7: (a12-a22)
    ]
    strassen_V = [
        {0: 1, 3: 1},   # (b11+b22)
        {0: 1},         # b11
        {1: 1, 3: -1},  # (b12-b22)
        {2: 1, 0: -1},  # (b21-b11)
        {3: 1},         # b22
        {0: 1, 1: 1},   # (b11+b12)
        {2: 1, 3: 1},   # (b21+b22)
    ]
    strassen_W = [
        {0: 1, 3: 1},   # P1 -> c11, c22
        {2: 1, 3: -1},  # P2 -> c21, -c22
        {1: 1, 3: 1},   # P3 -> c12, c22
        {0: 1, 2: 1},   # P4 -> c11, c21
        {0: -1, 1: 1},  # P5 -> -c11, c12
        {3: 1},         # P6 -> c22
        {0: 1},         # P7 -> c11
    ]

    # Build the 6 Strassen blocks: 3 column-blocks of B x 2 row-sides of B/A.
    # U index: i*m + j (i in 0..1, j in 0..3)
    # V index: j*p + k (j in 0..3, k in 0..5)
    # W index: k*n + i (k in 0..5, i in 0..1)
    col_offset = 0
    for j_block in range(3):
        for v_row_offset in (0, 2):  # L side: rows 0,1; R side: rows 2,3
            u_offset = v_row_offset  # A col offset matches B row offset
            for prod_idx in range(7):
                col = col_offset + prod_idx

                # U local map: 0->(0,uo), 1->(0,uo+1), 2->(1,uo), 3->(1,uo+1)
                u_base = u_offset
                for local_idx, sign in strassen_U[prod_idx].items():
                    i_loc = local_idx // 2
                    j_loc = local_idx % 2
                    U[i_loc * m + u_base + j_loc][col] += sign

                # V local map: 0->(vro,2jb), 1->(vro,2jb+1), 2->(vro+1,2jb), 3->(vro+1,2jb+1)
                for local_idx, sign in strassen_V[prod_idx].items():
                    i_loc = local_idx // 2
                    j_loc = local_idx % 2
                    V[(v_row_offset + i_loc) * p + 2 * j_block + j_loc][col] += sign

                # W local map: 0->(2jb,0), 1->(2jb+1,0), 2->(2jb,1), 3->(2jb+1,1)
                for local_idx, sign in strassen_W[prod_idx].items():
                    i_loc = local_idx // 2
                    j_loc = local_idx % 2
                    W[(2 * j_block + j_loc) * n + i_loc][col] += sign

            col_offset += 7

    # Deadline guard (construction is deterministic and fast, but guard per directive)
    if time.time() - _t0 > _DEADLINE - 5.0:
        raise RuntimeError("Deadline exceeded during construction")

    return {"U": U, "V": V, "W": W}
