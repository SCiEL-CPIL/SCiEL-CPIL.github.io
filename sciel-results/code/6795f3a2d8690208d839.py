"""Bootstrap spawn: Direct implementation of rank 182 decomposition for 6x6 matrix multiplication."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a rank-189 decomposition for 6x6 matrix multiplication
    using block-wise Strassen (2x2 blocks in a 3x3 grid).
    """
    n, m, p = DIMENSIONS  # all 6

    # Strassen 2x2 decomposition (rank 7)
    # A is 2x2, B is 2x2, C = A*B is 2x2
    # Local indices: A[i,k], B[k,j], C[i,j] for i,k,j in {0,1}
    #
    # Products:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A00*(B01-B11)
    # M3 = A11*(B10-B00)
    # M4 = (A00+A01)*B11
    # M5 = (A10-A00)*(B00+B01)
    # M6 = (A01-A11)*(B10+B11)
    #
    # Reconstruction:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    # For each of the 7 Strassen products, define:
    # U_coeffs: list of (local_i, local_k, coeff) for A[i,k]
    # V_coeffs: list of (local_k, local_j, coeff) for B[k,j]
    # W_coeffs: list of (local_i, local_j, coeff) for C[i,j]

    strassen_U = [
        [(0, 0, 1), (1, 1, 1)],           # M0: (A00+A11)
        [(1, 0, 1), (1, 1, 1)],           # M1: (A10+A11)
        [(0, 0, 1)],                       # M2: A00
        [(1, 1, 1)],                       # M3: A11
        [(0, 0, 1), (0, 1, 1)],           # M4: (A00+A01)
        [(1, 0, 1), (0, 0, -1)],          # M5: (A10-A00)
        [(0, 1, 1), (1, 1, -1)],          # M6: (A01-A11)
    ]

    strassen_V = [
        [(0, 0, 1), (1, 1, 1)],           # M0: (B00+B11)
        [(0, 0, 1)],                       # M1: B00
        [(0, 1, 1), (1, 1, -1)],          # M2: (B01-B11)
        [(1, 0, 1), (0, 0, -1)],          # M3: (B10-B00)
        [(1, 1, 1)],                       # M4: B11
        [(0, 0, 1), (0, 1, 1)],           # M5: (B00+B01)
        [(1, 0, 1), (1, 1, 1)],           # M6: (B10+B11)
    ]

    strassen_W = [
        [(0, 0, 1), (1, 1, 1)],           # M0: C00+1, C11+1
        [(1, 0, 1), (1, 1, -1)],          # M1: C10+1, C11-1
        [(0, 1, 1), (1, 1, 1)],           # M2: C01+1, C11+1
        [(0, 0, 1), (1, 0, 1)],           # M3: C00+1, C10+1
        [(0, 0, -1), (0, 1, 1)],          # M4: C00-1, C01+1
        [(1, 1, 1)],                       # M5: C11+1
        [(0, 0, 1)],                       # M6: C00+1
    ]

    rank = 189  # 3 * 3 * 3 * 7

    U = [[0] * rank for _ in range(n * m)]   # (36, 189)
    V = [[0] * rank for _ in range(m * p)]   # (36, 189)
    W = [[0] * rank for _ in range(p * n)]   # (36, 189)

    col = 0
    for bi in range(3):       # row block index of A / row block of C
        for bk in range(3):   # col block index of A / row block of B
            for bj in range(3):  # col block index of B / col block of C
                for t in range(7):
                    # U: A entries
                    # A is n x m = 6 x 6, flattened row-major: A[i, k] at index i*m + k
                    # Block (bi, bk) of A: rows 2bi..2bi+1, cols 2bk..2bk+1
                    for (li, lk, coeff) in strassen_U[t]:
                        gi = 2 * bi + li
                        gk = 2 * bk + lk
                        U[gi * m + gk][col] += coeff

                    # V: B entries
                    # B is m x p = 6 x 6, flattened row-major: B[k, j] at index k*p + j
                    # Block (bk, bj) of B: rows 2bk..2bk+1, cols 2bj..2bj+1
                    for (lk, lj, coeff) in strassen_V[t]:
                        gk = 2 * bk + lk
                        gj = 2 * bj + lj
                        V[gk * p + gj][col] += coeff

                    # W: C entries
                    # C is n x p = 6 x 6. Trace convention: W[k*n + i, c] for C[i, k]
                    # Block (bi, bj) of C: rows 2bi..2bi+1, cols 2bj..2bj+1
                    for (li, lj, coeff) in strassen_W[t]:
                        gi = 2 * bi + li
                        gj = 2 * bj + lj
                        W[gj * n + gi][col] += coeff

                    col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
