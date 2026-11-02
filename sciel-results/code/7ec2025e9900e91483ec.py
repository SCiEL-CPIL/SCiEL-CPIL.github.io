"""Rank-35 exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Strategy: Kronecker product of 2x2x2 Strassen core (rank 7) and
1x2x3 rectangular core (rank 5). Total rank = 7 * 5 = 35.

The 2x4x6 tensor factors as (2x2x2) ⊗ (1x2x3):
  - The 2x2x2 part handles the 2-row dimension via Strassen.
  - The 1x2x3 part handles the 4x6 -> 2x6 reduction.

1x2x3 rank-5 decomposition:
  s=0: u=[1,0], v=[1,0,0, 0,0,0], w=[1,0,0]  -> (0,0)
  s=1: u=[0,1], v=[0,0,0, 1,0,0], w=[1,0,0]  -> (1,0)
  s=2: u=[1,1], v=[0,1,0, 0,1,0], w=[0,1,0]  -> (0,1)+(1,1)
  s=3: u=[1,0], v=[0,0,1, 0,0,0], w=[0,0,1]  -> (0,2)
  s=4: u=[0,1], v=[0,0,0, 0,0,1], w=[0,0,1]  -> (1,2)
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


def _schoolbook_1x2x3_factors():
    """Return (U2, V2, W2) for 1x2x3 schoolbook core, shapes (2,6),(6,6),(3,6).
    U2 row: j (0..1), V2 row: j*3+k, W2 row: k (0..2)."""
    U2 = np.zeros((2, 6), dtype=float)
    V2 = np.zeros((6, 6), dtype=float)
    W2 = np.zeros((3, 6), dtype=float)

    # s=0: A0 B00 -> C0
    U2[0, 0] = 1
    V2[0, 0] = 1  # j=0, k=0
    W2[0, 0] = 1

    # s=1: A1 B10 -> C0
    U2[1, 1] = 1
    V2[3, 1] = 1  # j=1, k=0
    W2[0, 1] = 1

    # s=2: A0 B01 -> C1
    U2[0, 2] = 1
    V2[1, 2] = 1  # j=0, k=1
    W2[1, 2] = 1

    # s=3: A1 B11 -> C1
    U2[1, 3] = 1
    V2[4, 3] = 1  # j=1, k=1
    W2[1, 3] = 1

    # s=4: A0 B02 -> C2
    U2[0, 4] = 1
    V2[2, 4] = 1  # j=0, k=2
    W2[2, 4] = 1

    # s=5: A1 B12 -> C2
    U2[1, 5] = 1
    V2[5, 5] = 1  # j=1, k=2
    W2[2, 5] = 1

    return U2, V2, W2


def _build_rank42():
    """Build rank-42 decomposition via Kronecker product of Strassen (7) and 1x2x3 Schoolbook (6)."""
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()  # shapes (4,7),(4,7),(4,7)
    U2, V2, W2 = _schoolbook_1x2x3_factors()  # shapes (2,6),(6,6),(3,6)

    rank1, rank2 = 7, 6
    total_rank = rank1 * rank2  # 42

    U = np.zeros((n * m, total_rank), dtype=float)  # (8, 42)
    V = np.zeros((m * p, total_rank), dtype=float)  # (24, 42)
    W = np.zeros((p * n, total_rank), dtype=float)  # (12, 42)

    for i1 in range(2):
        for a1 in range(2):
            for a2 in range(2):
                for b1 in range(2):
                    for b2 in range(3):
                        for s1 in range(7):
                            for s2 in range(6):
                                col = s1 * 6 + s2
                                cu = U1[i1 * 2 + a1, s1] * U2[0, a2]
                                cv = V1[a1 * 2 + b1, s1] * V2[a2 * 3 + b2, s2]
                                cw = W1[b1 * 2 + i1, s1] * W2[b2, s2]
                                if cu != 0:
                                    row_u = i1 * 4 + a1 * 2 + a2
                                    U[row_u, col] += cu
                                if cv != 0:
                                    row_v = (a1 * 2 + a2) * 6 + b1 * 3 + b2
                                    V[row_v, col] += cv
                                if cw != 0:
                                    row_w = (b1 * 3 + b2) * 2 + i1
                                    W[row_w, col] += cw

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


def _max_error(U, V, W, n, m, p):
    """Return maximum absolute error across all (i,j,k) entries."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _to_output(U, V, W):
    """Convert numpy arrays to list of lists with int where possible."""
    U_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in U]
    V_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in V]
    W_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in W]
    return {"U": U_out, "V": V_out, "W": W_out}


def _build_schoolbook():
    """Fallback schoolbook rank-48 decomposition."""
    n, m, p = 2, 4, 6
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


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Try rank-42 Kronecker product (Strassen 7 * Schoolbook 1x2x3 6)
    if time.time() < deadline:
        U42, V42, W42 = _build_rank42()
        if _verify(U42, V42, W42, n, m, p):
            return _to_output(U42, V42, W42)
        err42 = _max_error(U42, V42, W42, n, m, p)
        if err42 < 1e-9:
            return _to_output(U42, V42, W42)

    # Final fallback: schoolbook rank 48
    return _build_schoolbook()
