"""Matrix multiplication 2x3x4 rank-minimal tensor decomposition."""
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin out of 150s

    n, m, p = 2, 3, 4  # A is n×m, B is m×p, C is n×p

    # Rank-16 construction:
    # For each k in 0..3 and each of 4 sign patterns, one term with value 1/2
    # U[i, j, r] = s_j (column sign), V[j, k, r] = 1, W[k, i, r] = 1
    #
    # Sign patterns (each a 3-tuple of ±1), chosen so that for each j,
    # the 4 patterns give {1,1,-1,-1} so the sum over r at each (i,j,k) = 1/2*2 = 1.
    #
    # 4 patterns:
    #   s0 = ( 1,  1,  1)
    #   s1 = ( 1,  1, -1)
    #   s2 = (-1, -1, -1)
    #   s3 = (-1, -1,  1)
    #
    # Check column 0: 1, 1, -1, -1 -> sum = 0, each pair of k gives sum 1
    # Actually let's verify: for fixed (i,j,k), sum over the 4 patterns of s_j = ?
    # j=0: 1+1+(-1)+(-1) = 0. That's wrong - we need sum = 1.
    #
    # Let me reconsider. We need: for each (i, j, k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # If V=1, W=1 for all: sum_r U[i,j,r] = 1 for each (i,j)
    # With 16 terms: 4 per k. For fixed (i,j), the 16 values U[i,j,r] must sum to 1.
    # Since 4 terms have V[j,k,r]=1 for each k (one per pattern), and 0 for other k's...
    # Wait, V[j,k,r] is 1 only for the specific k of term r.
    #
    # Let me redo: term r is associated with a specific (k, pattern).
    # For term r with k_r and pattern s_r:
    #   U[i, j, r] = s_r[j]
    #   V[j, k, r] = 1 if k == k_r, else 0
    #   W[k, i, r] = 1 if k == k_r, else 0
    #
    # So for fixed (i, j, k): only terms with k_r == k contribute.
    # There are 4 such terms (one per pattern).
    # Sum = (1/2) * sum_{4 patterns} s_r[j]
    # We need this = 1, so sum of 4 patterns at position j must be 2.
    #
    # Choose 4 patterns from {±1}^3 such that for each j, sum = 2:
    #   All +1: (1,1,1) -> contributes 1,1,1
    #   Need 3 more patterns where each column sums to 1.
    #   Try: (1,1,-1), (1,-1,1), (-1,1,1)
    #   Column 0: 1+1+1+(-1) = 2 ✓
    #   Column 1: 1+1+(-1)+1 = 2 ✓
    #   Column 2: 1+(-1)+1+1 = 2 ✓
    #
    # So patterns: (1,1,1), (1,1,-1), (1,-1,1), (-1,1,1)
    # Sum per column = 2, times 1/2 = 1. Correct!

    # 4 sign patterns, each a tuple of 3 signs
    patterns = [
        ( 1,  1,  1),
        ( 1,  1, -1),
        ( 1, -1,  1),
        (-1,  1,  1),
    ]

    rank = p * len(patterns)  # 4 * 4 = 16

    # Build matrices
    U = [[0] * rank for _ in range(n * m)]  # 6 x 16
    V = [[0] * rank for _ in range(m * p)]  # 12 x 16
    W = [[0] * rank for _ in range(p * n)]  # 8 x 16

    r = 0
    for k in range(p):
        for pat in patterns:
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = pat[j] / 2.0
            for j in range(m):
                V[j * p + k][r] = 1
            for i in range(n):
                W[k * n + i][r] = 1
            r += 1

    # Quick verification
    # Check that for each (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    # Fallback to rank-24 trivial solution
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj][col] = 1
                                V[jj * p + kk][col] = 1
                                W[kk * n + ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
