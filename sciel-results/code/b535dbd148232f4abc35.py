"""Rank-minimal exact decomposition for 2x2 * 2x12 matrix multiplication.

Strategy: the 2x12 product splits into six independent 2x2x2 blocks
(the 12 columns of the right matrix form six 2-column groups). Each
2x2x2 block is decomposed with the rank-7 Strassen identity, so the
total rank is 6 * 7 = 42 (versus the schoolbook 48).

Index conventions (matching the task contract):
  A is 2x2,  entry A[i,j] at row i*m + j  (m=2)
  B is 2x12, entry B[j,k] at row j*p + k  (p=12)
  C is 2x12, entry C[i,k] at row k*n + i  (n=2)  -- "trace convention"
For block b, the two output columns are k = 2b + kk (kk in {0,1}).
"""
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before the 150s kill

    n, m, p = 2, 2, 12
    nblocks = p // 2  # 6 independent 2x2x2 blocks

    # Strassen rank-7 factors for a single 2x2x2 block.
    # U rows: A11, A12, A21, A22  (row index = i*m + j)
    # V rows: B11, B12, B21, B22  (row index = j*p + k)
    # W rows: C11, C12, C21, C22  (row index = k*n + i)
    # The 7 columns are the Strassen products M1..M7.
    U0 = [
        [1, 0, 1, 0, 1, 1, 0],  # A11: M1, M3, M5, M6
        [0, 0, 0, 0, 1, 0, 1],  # A12: M5, M7
        [0, 1, 0, 0, 0, 1, 0],  # A21: M2, M6
        [1, 1, 0, 1, 0, 0, 1],  # A22: M1, M2, M4, M7
    ]
    V0 = [
        [1, 1, 0, 1, 0, 1, 1],  # B11: M1, M2, M4, M6, M7
        [0, 0, 1, 1, 0, 1, 0],  # B12: M3, M4, M6
        [0, 0, 0, 0, 0, 0, 1],  # B21: M7
        [1, 0, 1, 0, 1, 0, 0],  # B22: M1, M3, M5
    ]
    W0 = [
        [1, 0, 0, 1, -1, 0, 1],  # C11: M1, M4, -M5, M7
        [0, 0, 1, 0, 1, 0, 0],   # C12: M3, M5
        [0, 1, 0, 1, 0, 0, 0],   # C21: M2, M4
        [1, -1, 1, 0, 0, 1, 0],  # C22: M1, -M2, M3, M6
    ]

    rank = nblocks * 7  # 42
    U = [[0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 24 x 42

    for b in range(nblocks):
        col0 = b * 7
        for r in range(4):          # r = i*m+j for U, j for V, i for W
            for c in range(7):      # c = Strassen term index
                kk = c % 2          # which output column within the block
                # U: A[i,j] row = i*m + j ; r is exactly i*m+j
                U[r][col0 + c] = U0[r][c]
                # V: B[j,k] row = j*p + k ; r = j, k = 2b + kk
                V[r * p + (2 * b + kk)][col0 + c] = V0[r][c]
                # W: C[i,k] row = k*n + i ; r = i, k = 2b + kk
                W[(2 * b + kk) * n + r][col0 + c] = W0[r][c]

    # Deadline guard: construction is instantaneous; keep guard for safety.
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
