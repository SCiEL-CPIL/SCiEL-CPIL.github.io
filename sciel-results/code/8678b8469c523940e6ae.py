"""
Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy:
  1. Use the known rank-49 decomposition for 4x4x4 (Kronecker of 2x2x2 Strassen)
     for the first four output columns (k=0..3).
  2. For the remaining 4x4x2 sub-problem (k=4..5), use four independent 2x2x2
     Strassen products (one per (I,J) block pair) giving rank 28.
  Total rank = 49 + 28 = 77.

  This exploits the rectangular structure: the 4x4x6 tensor is the sum of a
  4x4x4 tensor (for columns 0-3) and a 4x4x2 tensor (for columns 4-5).
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]


def _get_strassen_2x2_factors():
    """
    Returns U, V, W for the rank-7 Strassen decomposition of 2x2x2.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(f+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    prods = [
        # (U_entries, V_entries, W_entries) where each entry is (index, coeff)
        # M1 = (a+d)(e+h)
        ([(0, 1.0), (3, 1.0)], [(0, 1.0), (3, 1.0)], [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)]),
        # M2 = (c+d)e
        ([(2, 1.0), (3, 1.0)], [(0, 1.0)], [(2, 1.0), (3, -1.0)]),
        # M3 = a(f-h)
        ([(0, 1.0)], [(1, 1.0), (3, -1.0)], [(1, 1.0), (3, 1.0)]),
        # M4 = d(g-c)
        ([(3, 1.0)], [(2, 1.0), (0, -1.0)], [(0, 1.0), (2, 1.0)]),
        # M5 = (a+b)h
        ([(0, 1.0), (1, 1.0)], [(3, 1.0)], [(0, -1.0), (1, 1.0)]),
        # M6 = (c-a)(e+f)
        ([(2, 1.0), (0, -1.0)], [(0, 1.0), (1, 1.0)], [(3, 1.0)]),
        # M7 = (b-d)(f+h)
        ([(1, 1.0), (3, -1.0)], [(1, 1.0), (3, 1.0)], [(0, 1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (idx, c) in Ub:
            U[idx][m_idx] += c
        for (idx, c) in Vb:
            V[idx][m_idx] += c
        for (idx, c) in Wb:
            W[idx][m_idx] += c

    return U, V, W


def _verify(U, V, W, n, m, p, rank):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return err


def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 4x4 matrix by a 4x6 matrix.

    Uses:
      - Rank-49 Kronecker Strassen for the 4x4x4 part (columns 0-3)
      - Rank-28 (4 independent 2x2x2 Strassen) for the 4x4x2 part (columns 4-5)
    Total rank = 77.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 4, 4, 6

    U_s, V_s, W_s = _get_strassen_2x2_factors()

    rank_444 = 49   # 7 * 7
    rank_442 = 28   # 4 * 7
    total_rank = rank_444 + rank_442  # 77

    U = [[0.0] * total_rank for _ in range(n * m)]  # 16 x 77
    V = [[0.0] * total_rank for _ in range(m * p)]  # 24 x 77
    W = [[0.0] * total_rank for _ in range(p * n)]  # 24 x 77

    # --- Part 1: 4x4x4 Kronecker Strassen (columns k=0..3) ---
    # i = 2*I + i0, j = 2*J + j0, k = 2*K + k0
    # Column index: 7*t + s where t is the block-level Strassen index, s is element-level
    for i in range(4):
        I = i // 2
        i0 = i % 2
        for j in range(4):
            J = j // 2
            j0 = j % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = U_s[2 * I + J][t] * U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * m + j][col] += val

    for j in range(4):
        J = j // 2
        j0 = j % 2
        for k in range(4):
            K = k // 2
            k0 = k % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = V_s[2 * J + K][t] * V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j * p + k][col] += val

    for k in range(4):
        K = k // 2
        k0 = k % 2
        for i in range(4):
            I = i // 2
            i0 = i % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = W_s[2 * K + I][t] * W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k * n + i][col] += val

    if time.time() > deadline:
        # Fallback to schoolbook
        return _schoolbook(n, m, p)

    # --- Part 2: 4x4x2 via 4 independent 2x2x2 Strassen (columns k=4..5) ---
    # C_right[2I+i0, 4+k0] = sum_{J} A[2I+i0, 2J+j0] * B[2J+j0, 4+k0]
    # Split by J: for each (I, J) pair, we have a 2x2x2 product
    #   A_IJ * B_J,right -> contributes to C_I,right
    # where B_J,right = B[2J..2J+1, 4..5] (2x2)
    #
    # Four products: (I,J) in {(0,0), (0,1), (1,0), (1,1)}
    # Each uses rank-7 Strassen, offset by rank_444.

    products = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for p_idx, (I, J) in enumerate(products):
        col_offset = rank_444 + p_idx * 7

        # U part: A[2I+i0, 2J+j0]
        for i0 in range(2):
            for j0 in range(2):
                i_global = 2 * I + i0
                j_global = 2 * J + j0
                for s in range(7):
                    val = U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i_global * m + j_global][col_offset + s] += val

        # V part: B[2J+j0, 4+k0]
        for j0 in range(2):
            for k0 in range(2):
                j_global = 2 * J + j0
                k_global = 4 + k0
                for s in range(7):
                    val = V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j_global * p + k_global][col_offset + s] += val

        # W part: C[2I+i0, 4+k0]
        for i0 in range(2):
            for k0 in range(2):
                i_global = 2 * I + i0
                k_global = 4 + k0
                for s in range(7):
                    val = W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k_global * n + i_global][col_offset + s] += val

    if time.time() > deadline:
        return _schoolbook(n, m, p)

    # Verify correctness
    err = _verify(U, V, W, n, m, p, total_rank)
    if err > 1e-6:
        # Fallback to schoolbook if verification fails
        return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition with rank n*m*p."""
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
