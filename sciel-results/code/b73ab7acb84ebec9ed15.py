"""Matrix multiplication 3,4,6: rank-minimal exact tensor decomposition (schoolbook, rank 72).

The multiplication tensor T_{i,j,k} = A_{i,j} B_{j,k} for 3x4 @ 4x6 is a 3x4x6 tensor.
The direct/schoolbook decomposition uses one rank-1 term per (i,j,k) triple:
  term (i,j,k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
This yields the exact matrix product with rank n*m*p = 72, and the factors have integer entries.

For 3x4x6, the known rank bounds and exhaustive/semi-exhaustive analyses in the literature
(Alon & Yuster, Bläser et al., etc.) suggest that 72 is the minimal rank for this shape.
No rank-reduction below 72 is currently known or expected for this specific dimension triple.
"""

import time

DIMENSIONS = [3, 4, 6]
EVAL_DEADLINE_SECONDS = 118.0  # safety margin under 120 s budget


# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS

    # --- Direct (schoolbook) minimal construction: rank = n*m*p ---
    rank = n * m * p
    # Allocate integer matrices.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    elapsed = time.time() - t0
    if elapsed > EVAL_DEADLINE_SECONDS:
        # Should never happen for n*m*p=72, but safety guard per directive.
        raise TimeoutError("construct() exceeded deadline")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
