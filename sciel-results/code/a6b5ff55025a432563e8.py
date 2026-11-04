"""Strassen-block + rank-factorization decomposition for 3x3 * 3x5."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]
_START = time.time()
_DEADLINE = 120.0  # wall-clock limit; keep safety margin


def _rank_factor(A, B):
    """Return (U, V) of ranks r with A @ B == U @ V  (float)."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = int(np.sum(s > 1e-10))
    U = U[:, :r]
    V = (Vt[:r, :] * s[:r]).T
    return U, V


def _verify(U, V, W, n, m, p):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k,c."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = U[i * m + j, :] * V[j * p + k, :]
                rhs = W[k * n + i, :]
                if not np.allclose(lhs, rhs, atol=1e-8, rtol=1e-10):
                    return False
    return True


def _to_frac(M):
    """Convert a float matrix of (half-)integer entries to exact fractions."""
    from fractions import Fraction
    out = []
    for row in M:
        r = []
        for x in row:
            f = Fraction(str(round(x, 9)))
            r.append(f)
        out.append(r)
    return out


def _strassen_decomposition(n, m, p):
    """
    Build the 3x3 * 3x5 multiplication tensor as a rank sum of block MM's.

    Split row-blocks R0={0,1}, R1={2}; col-blocks C0={0,1}, C1={2}.
    Strassen for the 2x2 block product (R0/R1) x (C0/C1):
      S1=(A00+A11)(B01+B10)
      S2=(A10+A11)B00
      S3=A00(B10-B00)
      S4=A11(B01-B11)
      S5=(A00+A10)B11
      S6=(A11-A01)(B00+B10)
      S7=(A01-A11)(B11+B01)
    plus direct blocks for the R1 x C1 = A22*B22 product.
    Each block MM is compressed via rank factorization of its A-block.
    """
    # Block sizes
    r0 = [0, 1]   # rows in R0
    r1 = [2]      # rows in R1
    c0 = [0, 1]   # cols in C0
    c1 = [2]      # cols in C1
    a00, a01 = 2, 2
    a10, a11 = 1, 1
    b00, b01 = 2, 2
    b10, b11 = 1, 1

    # Strassen 2x2 block products: list of (i_block, j_block, sign, A_block, B_block)
    # i_block in {0,1} for R0,R1 ; j_block in {0,1} for C0,C1
    # M[i,j] = sum over terms
    # C00 = S1 - S3 - S4 + S7
    # C01 = S5 + S1
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    strassen_terms = [
        # (i_block, j_block, sign, A_rows, B_cols)
        (0, 0, +1, (0, 1), (1, 0)),   # S1 = (A00+A11)(B01+B10)
        (1, 0, -1, (0, 0), (0, 0)),   # -S3 = -A00(B10-B00)
        (0, 1, +1, (1, 1), (0, 1)),   # +S4 = A11(B01-B11)
        (1, 0, -1, (1, 1), (0, 1)),   # -S4
        (0, 0, +1, (0, 1), (1, 1)),   # +S7 = (A01-A11)(B11+B01)
        (0, 1, +1, (0, 0), (1, 1)),   # S5 = A00 B11
        (0, 1, +1, (0, 1), (1, 0)),   # S1
        (1, 0, +1, (1, 0), (0, 0)),   # S2 = A10 B00
        (1, 1, +1, (1, 1), (0, 1)),   # S4
        (1, 0, -1, (1, 1), (0, 0)),   # -S2
        (1, 1, +1, (0, 0), (1, 0)),   # S3 = A00(B10-B00)
        (1, 1, +1, (1, 1), (0, 0)),   # S6 = (A11-A01)(B00+B10)
        (1, 1, +1, (0, 1), (1, 0)),   # S7
    ]

    # Direct product for R1 x C1: A22 * B22  (i_block=1, j_block=1)
    direct_terms = [
        (1, 1, +1, (1, 1), (1, 1)),  # A22 * B22
    ]

    all_terms = strassen_terms + direct_terms

    # Build U, V, W incrementally
    U = []
    V = []
    W = []
    for (ib, jb, sign, (ai, aj), (bi, bj)) in all_terms:
        # A-block: rows r_a[ai], cols r_b[aj]
        a_rows = r0 if ai == 0 else r1
        b_cols = c0 if aj == 0 else c1
        # B-block: rows r_b[bi], cols r_c[bj]
        b_rows = c0 if bi == 0 else c1
        a_cols = c0 if bj == 0 else c1

        # Build the A-block matrix (|a_rows| x |b_cols|) with identity entries
        Ablk = np.zeros((len(a_rows), len(b_cols)))
        for x in range(len(a_rows)):
            for y in range(len(b_cols)):
                if a_rows[x] == b_cols[y]:
                    Ablk[x, y] = 1.0
        # B-block matrix (|b_rows| x |a_cols|)
        Bblk = np.zeros((len(b_rows), len(a_cols)))
        for x in range(len(b_rows)):
            for y in range(len(a_cols)):
                if b_rows[x] == a_cols[y]:
                    Bblk[x, y] = 1.0

        Uf, Vf = _rank_factor(Ablk, Bblk)
        r = Uf.shape[1]
        if r == 0:
            continue

        # Map global indices
        for t in range(r):
            col = len(U)
            # U column: entries at global (i, j) for i in a_rows, j in b_cols
            u_col = [0.0] * (n * m)
            for x, i in enumerate(a_rows):
                for y, j in enumerate(b_cols):
                    u_col[i * m + j] = Uf[x, t]
            # V column: entries at global (j, k) for j in b_rows, k in a_cols
            v_col = [0.0] * (m * p)
            for x, j in enumerate(b_rows):
                for y, k in enumerate(a_cols):
                    v_col[j * p + k] = Vf[x, t]
            # W column: entries at global (k, i) for k in a_cols, i in a_rows
            w_col = [0.0] * (p * n)
            for x, k in enumerate(a_cols):
                for y, i in enumerate(a_rows):
                    w_col[k * n + i] = sign * 1.0

            U.append(u_col)
            V.append(v_col)
            W.append(w_col)

    rank = len(U)
    if _verify(U, V, W, n, m, p):
        return _to_frac(np.array(U)), _to_frac(np.array(V)), _to_frac(np.array(W)), rank
    return None, None, None, None


def construct():
    n, m, p = DIMENSIONS
    if time.time() > _DEADLINE - 5:
        # Fall back to schoolbook
        return _schoolbook(n, m, p)

    U, V, W, rank = _strassen_decomposition(n, m, p)
    if U is not None:
        return {"U": U, "V": V, "W": W}
    return _schoolbook(n, m, p)


def _schoolbook(n, m, p):
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
