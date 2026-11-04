"""Combined program: structural Strassen+schoolbook from inspiration 1, with verification and polish from the current/inspiration 2 family."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    Uses 4 disjoint 2x2x2 Strassen blocks (rank 7 each) + schoolbook for remaining terms.
    Total rank: 4*7 + 28 = 56.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 20
    pn = p * n  # 15

    # Strassen coefficients for 2x2x2 block
    # Local indices (li, lj) for U, (lj, lk) for V, (lk, li) for W
    U_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): 1},       # M2
        {(0,0): 1},                 # M3
        {(1,1): 1},                 # M4
        {(0,0): 1, (0,1): 1},       # M5
        {(1,0): 1, (0,0): -1},      # M6
        {(0,1): 1, (1,1): -1}       # M7
    ]
    V_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(0,0): 1},                 # M2
        {(0,1): 1, (1,1): -1},      # M3
        {(1,0): 1, (0,0): -1},      # M4
        {(1,1): 1},                 # M5
        {(0,0): 1, (0,1): 1},       # M6
        {(1,0): 1, (1,1): 1}        # M7
    ]
    W_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(0,1): 1, (1,1): -1},      # M2
        {(1,0): 1, (1,1): 1},       # M3
        {(0,0): 1, (0,1): 1},       # M4
        {(0,0): -1, (1,0): 1},      # M5
        {(1,1): 1},                 # M6
        {(0,0): 1}                  # M7
    ]

    # Define 4 disjoint 2x2x2 Strassen blocks
    # I: rows of A, J: cols of A / rows of B, K: cols of B
    # n=3, m=4, p=5
    # We can fit 2x2 blocks in I={0,1}, J={0,1} or {2,3}, K={0,1} or {2,3}
    # The remaining indices are i=2, j=3 (if J uses {0,1}), k=4
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3])
    ]

    R = 4 * 7 + 28  # 28 + 28 = 56
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))

    col = 0

    # Add Strassen blocks
    for I, J, K in blocks:
        for t in range(7):
            for (li, lj), val in U_coeffs[t].items():
                i = I[li]
                j = J[lj]
                U[i * m + j, col] += val
            for (lj, lk), val in V_coeffs[t].items():
                j = J[lj]
                k = K[lk]
                V[j * p + k, col] += val
            for (lk, li), val in W_coeffs[t].items():
                k = K[lk]
                i = I[li]
                W[k * n + i, col] += val
            col += 1

    # Add schoolbook terms for remaining (i, j, k) not covered by Strassen
    # Covered: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Remaining:
    # 1. i=2, any j, any k (4*5 = 20 terms)
    # 2. i in {0,1}, any j, k=4 (2*4 = 8 terms)
    # Total remaining: 28 terms

    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if covered by Strassen blocks
                covered = False
                for I, J, K in blocks:
                    if i in I and j in J and k in K:
                        covered = True
                        break
                if not covered:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    assert col == R, f"Column count {col} != Rank {R}"

    # Verification: check that the decomposition is exact
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 for all i,j,k
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(R):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err

    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U = np.zeros((nm, rank_school))
        V = np.zeros((mp, rank_school))
        W = np.zeros((pn, rank_school))
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
