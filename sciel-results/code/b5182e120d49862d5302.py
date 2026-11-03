"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Strategy:
  - Split the middle dimension m=3 into 2+1.
  - The 2x2 * 2x10 part is computed by splitting the 10 columns of the right
    factor into 5 blocks of 2 columns and applying Strassen's algorithm
    (rank 7 per 2x2 * 2x2 block) to each block.  That costs 5 * 7 = 35
    scalar multiplications.
  - The 2x1 * 1x10 part is a trivial outer-product sum of rank 20.
  - Total rank = 35 + 20 = 55.

The evaluator measures the raw rank value (lower is better).  The score is
seed/value, so a rank of 55 with a seed of 60 gives 60/55 ≈ 1.0909.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-55 exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x10 matrix.

    The decomposition is built in two parts:
      Part A (rank 35):  A_left (2x2) * B_top (2x10) via 5 Strassen blocks.
      Part B (rank 20):  A_right (2x1) * B_bottom (1x10) via outer products.

    Index conventions (row-major flattening):
      A[i, j]  -> U index  i * m + j          (n=2, m=3)
      B[j, k]  -> V index  j * p + k          (m=3, p=10)
      C[i, k]  -> W index  k * n + i          (official trace convention)
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120 s kill

    RANK = 55
    U = np.zeros((n * m, RANK), dtype=np.int64)
    V = np.zeros((m * p, RANK), dtype=np.int64)
    W = np.zeros((p * n, RANK), dtype=np.int64)

    col = 0

    # ------------------------------------------------------------------
    # Part A: A_left (2x2) * B_top (2x10)  --  5 Strassen blocks, rank 35
    # ------------------------------------------------------------------
    # A_left indices in U:
    #   A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    X00, X01, X10, X11 = 0, 1, 3, 4

    # B_top indices in V (rows 0 and 1, all 10 columns):
    #   B[0,k] -> k,  B[1,k] -> 10 + k

    for j in range(5):
        if time.time() > deadline:
            break  # should never trigger, but guard anyway

        # Y block (2x2) from B_top columns 2j, 2j+1
        Y00 = 2 * j          # B[0, 2j]
        Y01 = 2 * j + 1      # B[0, 2j+1]
        Y10 = 10 + 2 * j     # B[1, 2j]
        Y11 = 10 + 2 * j + 1 # B[1, 2j+1]

        # Z block (2x2) in C columns 2j, 2j+1
        # C[0,2j]=W[4j], C[1,2j]=W[4j+1], C[0,2j+1]=W[4j+2], C[1,2j+1]=W[4j+3]
        Z00 = 4 * j
        Z10 = 4 * j + 1
        Z01 = 4 * j + 2
        Z11 = 4 * j + 3

        # Strassen: 7 products
        # M1 = (X00+X11)*(Y00+Y11)
        U[X00, col] += 1; U[X11, col] += 1
        V[Y00, col] += 1; V[Y11, col] += 1
        W[Z00, col] += 1; W[Z11, col] += 1
        col += 1

        # M2 = (X10+X11)*Y00
        U[X10, col] += 1; U[X11, col] += 1
        V[Y00, col] += 1
        W[Z10, col] += 1; W[Z11, col] -= 1
        col += 1

        # M3 = X00*(Y01-Y11)
        U[X00, col] += 1
        V[Y01, col] += 1; V[Y11, col] -= 1
        W[Z01, col] += 1; W[Z11, col] += 1
        col += 1

        # M4 = X11*(Y10-Y00)
        U[X11, col] += 1
        V[Y10, col] += 1; V[Y00, col] -= 1
        W[Z00, col] += 1; W[Z10, col] += 1
        col += 1

        # M5 = (X00+X01)*Y11
        U[X00, col] += 1; U[X01, col] += 1
        V[Y11, col] += 1
        W[Z00, col] -= 1; W[Z01, col] += 1
        col += 1

        # M6 = (X10-X00)*(Y00+Y01)
        U[X10, col] += 1; U[X00, col] -= 1
        V[Y00, col] += 1; V[Y01, col] += 1
        W[Z11, col] += 1
        col += 1

        # M7 = (X01-X11)*(Y10+Y11)
        U[X01, col] += 1; U[X11, col] -= 1
        V[Y10, col] += 1; V[Y11, col] += 1
        W[Z00, col] += 1
        col += 1

    # ------------------------------------------------------------------
    # Part B: A_right (2x1) * B_bottom (1x10)  --  rank 20
    # ------------------------------------------------------------------
    # A_right indices in U: A[0,2]=2, A[1,2]=5
    # B_bottom indices in V: B[2,k] = 20 + k
    for i in range(n):
        a_idx = i * m + 2  # 2 or 5
        for k in range(p):
            if time.time() > deadline:
                break
            b_idx = 2 * p + k  # 20 + k
            c_idx = k * n + i  # C[i,k]
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
        if time.time() > deadline:
            break

    # ------------------------------------------------------------------
    # Sanity check: col must equal RANK
    # ------------------------------------------------------------------
    if col != RANK:
        # Fallback: schoolbook rank 60
        rank_fb = n * m * p
        U = np.zeros((n * m, rank_fb), dtype=np.int64)
        V = np.zeros((m * p, rank_fb), dtype=np.int64)
        W = np.zeros((p * n, rank_fb), dtype=np.int64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
