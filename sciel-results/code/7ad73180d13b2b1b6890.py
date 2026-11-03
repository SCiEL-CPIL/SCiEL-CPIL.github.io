"""Combined program: block Strassen decomposition with exact verification and time guard.

Strategy: decompose 2x3x7 matrix multiplication into blocks where Strassen's
rank-7 algorithm applies to 2x2x2 sub-blocks, reducing total rank from 42
(schoolbook) to 39.
"""
import time

DIMENSIONS = [2, 3, 7]

# Hard wall-clock safety margin (seconds). Evaluator kills at 150 s.
_DEADLINE_SEC = 145.0


def _elapsed_since(start):
    return time.monotonic() - start


def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication. Rank 7.

    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r] for i,j,k in {0,1}.
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    # M1 = (A00 + A11)(B00 + B11)
    # C00 += 1, C11 += 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (A10 + A11)B00
    # C10 += 1, C11 -= 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    # M3 = A00(B10 - B11)
    # C01 += 1, C11 += 1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4 = A11(B01 - B00)
    # C00 += 1, C10 += 1
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5 = (A00 + A01)B11
    # C00 -= 1, C01 += 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6 = (A10 - A00)(B00 + B01)
    # C11 += 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (A01 - A11)(B10 + B11)
    # C00 += 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W, 7


def _block_decomposition(n, m, p):
    """Construct rank-39 decomposition for 2x3x7 using Strassen blocks.

    Block structure:
    - j=0: schoolbook 2x1x7, rank 14
    - j=1,2: split p into pairs (0,1),(2,3),(4,5) using Strassen 2x2x2,
      plus singleton k=6 using schoolbook 2x2x1.
      Total for j=1,2: 3*7 + 4 = 25
    Grand total: 39
    """
    # Precompute Strassen factors
    sU, sV, sW, srank = _strassen_2x2x2()

    total_rank = 14 + 3 * 7 + 4  # 39

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    # Block 1: j=0, schoolbook 2x1x7 (n=2, m_block=1, p=7)
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Block 2: j=1,2 (m_block=2), split p into pairs
    # p-pairs: (0,1), (2,3), (4,5)
    p_pairs = [(0, 1), (2, 3), (4, 5)]

    for (k0, k1) in p_pairs:
        # Strassen 2x2x2 for j in {1,2}, k in {k0, k1}
        for r in range(srank):
            # U: rows i*m + j for i in {0,1}, j in {1,2}
            for i in range(2):
                for j in range(2):
                    U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
            # V: rows j*p + k for j in {1,2}, k in {k0, k1}
            for j in range(2):
                for kk in range(2):
                    V[(1 + j) * p + (k0 + kk)][col] += sV[j * 2 + kk][r]
            # W: rows k*n + i for k in {k0, k1}, i in {0,1}
            for kk in range(2):
                for i in range(2):
                    W[(k0 + kk) * n + i][col] += sW[kk * 2 + i][r]
            col += 1

    # Block 3: j=1,2, k=6 (singleton), schoolbook 2x2x1
    for i in range(2):
        for j in range(2):
            U[i * m + (1 + j)][col] = 1
            V[(1 + j) * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1

    assert col == total_rank, f"Column count mismatch: {col} != {total_rank}"

    return {"U": U, "V": V, "W": W}


def _verify_exact(n, m, p, U, V, W):
    """Check that sum_c U[a,c]*V[b,c]*W[d,c] == 1 exactly when
    a = i*m+j, b = j*p+k, d = k*n+i, and 0 otherwise.
    Uses pure integer arithmetic, so the check is exact.
    """
    rank = len(U[0])
    for a in range(n * m):
        for b in range(m * p):
            for d in range(p * n):
                total = 0
                for c in range(rank):
                    total += U[a][c] * V[b][c] * W[d][c]
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(d, n)
                expected = 1 if (j == j2 and k == k2 and i == i2) else 0
                if total != expected:
                    return False
    return True


def _schoolbook_decomposition(n, m, p):
    """Exact rank n*m*p decomposition with 0/1 integer entries (fallback)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def construct():
    start = time.monotonic()
    n, m, p = DIMENSIONS

    # Build the block Strassen decomposition (rank 39 for 2x3x7)
    result = _block_decomposition(n, m, p)

    # Safety: if we are running out of time, return the result immediately
    if _elapsed_since(start) > _DEADLINE_SEC:
        return result

    # Exact verification (integer arithmetic)
    ok = _verify_exact(n, m, p, result["U"], result["V"], result["W"])

    if not ok:
        # Fallback to schoolbook if verification fails (should never happen)
        result = _schoolbook_decomposition(n, m, p)

    return result
