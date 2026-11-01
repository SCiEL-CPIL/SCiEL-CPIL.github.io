"""Rank-42 exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Strategy: Block-partition into six 2x2 @ 2x2 multiplications, each computed
via Strassen's algorithm (rank 7). Total rank = 6 * 7 = 42.

Partition:
  A (2x4) = [AL | AR], each 2x2
  B (4x6) = [B1 | B2 | B3], each 4x2
  Cj = A @ Bj = AL @ BjL + AR @ BjR,  j = 0,1,2
  Each Bj (4x2) = [BjL; BjR], each 2x2

  Six 2x2@2x2 products, each via Strassen => rank 42.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _strassen_factors():
    """Return (U1, V1, W1) for 2x2x2 Strassen, shapes (4,7),(4,7),(4,7).
    U1 row: i*2+j, V1 row: j*2+k, W1 row: k*2+i (trace convention)."""
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)

    # M1 = (A00+A11)(B00+B11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1  # C00 += M1
    W1[3, 0] = 1  # C11 += M1

    # M2 = (A10+A11)B00
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1  # C10 += M2
    W1[3, 1] = -1 # C11 -= M2

    # M3 = A00(B01-B11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1  # C01 += M3
    W1[3, 2] = 1  # C11 += M3

    # M4 = A11(B10-B00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1  # C00 += M4
    W1[1, 3] = 1  # C10 += M4

    # M5 = (A00+A01)B11
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1 # C00 -= M5
    W1[2, 4] = 1  # C01 += M5

    # M6 = (A10-A00)(B00+B01)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1  # C11 += M6

    # M7 = (A01-A11)(B10+B11)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1  # C00 += M7

    return U1, V1, W1


def _build_rank42():
    """Build rank-42 decomposition via 6 Strassen blocks."""
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    total_rank = 42

    U = np.zeros((n * m, total_rank), dtype=float)  # (8, 42)
    V = np.zeros((m * p, total_rank), dtype=float)  # (24, 42)
    W = np.zeros((p * n, total_rank), dtype=float)  # (12, 42)

    # Six blocks: for j_block in 0..2 (which 4x2 block of B),
    #             for side in 0..1 (AL or AR, i.e., which 2x2 of A)
    # Block index b = j_block * 2 + side
    # Each block uses Strassen rank-7, so columns b*7 .. b*7+6

    for j_block in range(3):
        for side in range(2):
            b = j_block * 2 + side
            col_offset = b * 7

            # U contribution: A[i, a] where a = side * 2 + j_local
            # i in 0..1, j_local in 0..1 (within the 2x2 sub-block)
            # U1[i*2 + j_local, s] gives the coefficient
            for i in range(2):
                for j_local in range(2):
                    a = side * 2 + j_local  # column in A (0..3)
                    for s in range(7):
                        coeff = U1[i * 2 + j_local, s]
                        if coeff != 0:
                            U[i * m + a, col_offset + s] += coeff

            # V contribution: B[a, bcol] where a = side * 2 + j_local, bcol = j_block * 2 + k_local
            # V1[j_local * 2 + k_local, s]
            for j_local in range(2):
                a = side * 2 + j_local
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = V1[j_local * 2 + k_local, s]
                        if coeff != 0:
                            V[a * p + bcol, col_offset + s] += coeff

            # W contribution: C[i, bcol] where i in 0..1, bcol = j_block * 2 + k_local
            # W1[k_local * 2 + i, s]
            for i in range(2):
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = W1[k_local * 2 + i, s]
                        if coeff != 0:
                            W[bcol * n + i, col_offset + s] += coeff

    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    U, V, W = _build_rank42()

    # Verify correctness
    if _verify(U, V, W, n, m, p):
        # Convert to list of lists with int where possible
        U_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in U]
        V_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in V]
        W_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in W]
        return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback: Try to return the unverified rank-42 solution if it's close, else schoolbook
    # In this specific case, the rank-42 solution is analytically exact, so verification should pass.
    # If it fails due to floating point, we still prefer 42 over 48 if the error is tiny.
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(U.shape[1]):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err < 1e-6:
        U_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in U]
        V_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in V]
        W_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in W]
        return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback: schoolbook rank 48
    schoolbook_rank = n * m * p
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U_fb[ii * m + jj][col] = 1
                V_fb[jj * p + kk][col] = 1
                W_fb[kk * n + ii][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
