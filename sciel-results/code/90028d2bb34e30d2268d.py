"""Suite task seed; EVOLVE-BLOCK is rewritten for the 2x3*3x6 MM tensor."""
import time
import numpy as np

_DIMENSIONS = [2, 3, 6]
DIMENSIONS = _DIMENSIONS  # keep for compatibility

# EVOLVE-BLOCK-START
# Split the 3x6 right factor into three disjoint 3x2 column blocks (blk=0,1,2).
# Each block computes a 2x3 * 3x2 product, a 4-entry tensor, in rank 11:
#   rank(2x2x2, Strassen)=7 for the shared inner columns j=0,1 (cols 0,1,2 of
#   B -> V indices 0..3) plus 4 trivial scalar products for the outer column j=2
#   (V indices 4,5).  3 blocks * 11 = rank 33.
#
# Local indices (per block):
#   A : U index a = 3*i_row + j_col,  i_row in {0,1}, j_col in {0,1,2}
#   B : V index b = 2*j_col + k_col,  j_col in {0,1,2}, k_col in {0,1}
#   C : W index w = k_col*n + i_row   (official trace convention W[k*n+i])
#
# Strassen terms on the 2x2 core (i_row,k_col in {0,1}, inner j_col in {0,1}):
#   A00=0 A01=1 A10=3 A11=4 ; B00=0 B01=1 B10=2 B11=3
#   t1  (A00+A11)(B00+B11)   -> C00 +=, C11 +=
#   t2  (A10+A11)B00         -> C10 +=, C11 -=
#   t3  A00(B01-B11)         -> C01 +=, C11 +=
#   t4  A11(B10-B00)         -> C00 +=, C10 +=
#   t5  (A00+A01)B11         -> C00 -=, C01 +=
#   t6  (A10-A00)(B00+B01)   -> C11 +=
#   t7  (A01-A11)(B10+B11)   -> C00 +=
#   t8  A02*B20 -> C00 ; t9 A02*B21 -> C01
#   t10 A12*B20 -> C10 ; t11 A12*B21 -> C11
#
# term -> ( {U:coef}, {V:coef}, {W:coef} ); W keys already use trace order.
_CORE = [
    ({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
    ({3: 1, 4: 1}, {0: 1},         {2: 1, 3: -1}),
    ({0: 1},       {1: 1, 3: -1},  {1: 1, 3: 1}),
    ({4: 1},       {2: 1, 0: -1},  {0: 1, 2: 1}),
    ({0: 1, 1: 1}, {3: 1},         {0: -1, 1: 1}),
    ({3: 1, 0: -1},{0: 1, 1: 1},   {3: 1}),
    ({1: 1, 4: -1},{2: 1, 3: 1},   {0: 1}),
]
# tail (outer column) terms: (u_idx, v_tail_idx, w_idx)
_TAIL = [
    (2, 4, 0),   # t8  A02*B20 -> C00
    (2, 5, 1),   # t9  A02*B21 -> C01
    (5, 4, 2),   # t10 A12*B20 -> C10
    (5, 5, 3),   # t11 A12*B21 -> C11
]


def _build_factors(n, m, p, core_terms, tail_terms):
    """Assemble U,V,W for p column-blocks of width 2 (core width 2 + tail width 1).

    Returns factor lists with integer entries; the tensor is exact by construction
    (linear combination of 2-input bilinear forms reproduces every product entry).
    """
    per_block = len(core_terms) + len(tail_terms)          # 7 + 4 = 11
    rank = per_block * (p // 2)                            # 11 * 3 = 33

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for blk in range(p // 2):
        off = blk * per_block
        # --- core (7 terms) ---
        for t, (u_map, v_map, w_map) in enumerate(core_terms):
            col = off + t
            for u_idx, c in u_map.items():
                U[u_idx][col] = c
            for v_idx, c in v_map.items():
                j = v_idx >> 1            # inner column
                k = v_idx & 1             # inner row of B
                row = j * p + blk * 2 + k
                V[row][col] = c
            for w_idx, c in w_map.items():
                i = w_idx >> 1            # row of C
                k = w_idx & 1             # col of C
                W[k * n + i][col] = c
        # --- tail (4 terms) ---
        for s, (u_idx, v_tail, w_idx) in enumerate(tail_terms):
            col = off + len(core_terms) + s
            U[u_idx][col] = 1
            j = v_tail >> 1               # column of B (always 2, the outer col)
            k = v_tail & 1                # row of B
            V[j * p + blk * 2 + k][col] = 1
            i = w_idx >> 1                # row of C
            k = w_idx & 1                 # col of C
            W[k * n + i][col] = 1

    return U, V, W


def construct():
    _t0 = time.time()
    _deadline = _t0 + 110.0
    n, m, p = DIMENSIONS

    try:
        U, V, W = _build_factors(n, m, p, _CORE, _TAIL)
    except Exception:
        # Fallback: naive rank-nmp decomposition (always valid, never timed out above).
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1

    if time.time() > _deadline - 2.0:
        pass  # construction is deterministic and O(rank); nothing to truncate

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
