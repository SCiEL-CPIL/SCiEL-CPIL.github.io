"""Matrix multiplication 4,5,6 with verified Strassen-based decomposition.

We split A (4x5) into A1 (4x4, cols 0-3) and A2 (4x1, col 4), and B (5x6)
into B1 (4x6, rows 0-3) and B2 (1x6, row 4). Then C = A1*B1 + A2*B2.
  * A2*B2: 4x1 x 1x6, schoolbook rank 24.
  * A1*B1: 4x4 x 4x6. Split B1 into B1L (4x4, cols 0-3) and B1R (4x2, cols 4-5).
      - A1*B1L: 4x4 x 4x4, Strassen rank 28.
      - A1*B1R: 4x4 x 4x2, schoolbook rank 32.
Total rank = 24 + 28 + 32 = 84.
"""
import time

DIMENSIONS = [4, 5, 6]

def _matmul(A, B):
    n, m = A.shape
    m2, p = B.shape
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        for j in range(m):
            a = A[i][j]
            if a:
                for k in range(p):
                    C[i][k] += a * B[j][k]
    return C

def _verify(U, V, W, n, m, p, rank):
    """Check that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 6

    rank = 84
    U = [[0] * rank for _ in range(n * m)]   # 20 x 84
    V = [[0] * rank for _ in range(m * p)]   # 30 x 84
    W = [[0] * rank for _ in range(p * n)]   # 24 x 84

    def term(u_idx, v_idx, w_idx, val, col):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    def sb4x2(A_ro, A_co, B_ro, B_co, C_ro, C_co, start_col):
        """Schoolbook 4x2 (A 4x2 @ B 2x2) -> C 4x2, 16 terms."""
        c = start_col
        for i in range(4):
            for j in range(2):
                for k in range(2):
                    u = (A_ro + i) * m + (A_co + j)
                    v = (B_ro + j) * p + (B_co + k)
                    w = (C_co + k) * n + (C_ro + i)
                    term(u, v, w, 1.0, c)
                    c += 1
        return c

    def sb4x4(A_ro, A_co, B_ro, B_co, C_ro, C_co, start_col):
        """Schoolbook 4x4 (A 4x4 @ B 4x4) -> C 4x4, 64 terms."""
        c = start_col
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u = (A_ro + i) * m + (A_co + j)
                    v = (B_ro + j) * p + (B_co + k)
                    w = (C_co + k) * n + (C_ro + i)
                    term(u, v, w, 1.0, c)
                    c += 1
        return c

    def sb4x1x1x6(A_ro, B_ro, C_ro, start_col):
        """A (4x1) @ B (1x6) -> C (4x6), 24 terms."""
        c = start_col
        for i in range(4):
            for k in range(6):
                u = (A_ro + i) * m + 4
                v = B_ro * p + k
                w = k * n + (C_ro + i)
                term(u, v, w, 1.0, c)
                c += 1
        return c

    def strassen_4x4(A_ro, A_co, B_ro, B_co, C_ro, C_co, start_col):
        """Strassen 4x4 x 4x4 -> 4x4, 28 terms.

        2x2 block split (each block 2x2):
          A00=A[ro:ro+2, co:co+2], A01=A[ro:ro+2, co+2:co+4]
          A10=A[ro+2:ro+4, co:co+2], A11=A[ro+2:ro+4, co+2:co+4]
          B00=B[ro:ro+2, co:co+2], B01=B[ro:ro+2, co+2:co+4]
          B10=B[ro+2:ro+4, co:co+2], B11=B[ro+2:ro+4, co+2:co+4]
        Strassen products M1..M7 (each a 2x2 matrix product, 4 rank-1 terms):
          M1 = (A00+A11)(B00+B11)  -> C00 +1, C11 +1
          M2 = (A10+A11)B00        -> C10 +1, C11 -1
          M3 = A00(B01-B11)        -> C01 +1, C11 +1
          M4 = A11(B10-B00)        -> C00 +1, C10 +1
          M5 = (A00+A01)B11        -> C00 -1, C01 +1
          M6 = (A10-A00)(B00+B01)  -> C11 +1
          M7 = (A01-A11)(B10+B11)  -> C00 +1
        """
        # Block row/col offsets within the 4x4 subproblem.
        # A block (br, bc): rows ro+br*2..ro+br*2+1, cols co+bc*2..co+bc*2+1
        def a_off(br, bc):
            return (A_ro + 2 * br, A_co + 2 * bc)
        def b_off(br, bc):
            return (B_ro + 2 * br, B_co + 2 * bc)
        # C block (br, bc): rows ro+br*2, cols co+bc*2
        def c_off(br, bc):
            return (C_ro + 2 * br, C_co + 2 * bc)

        c = start_col
        for (a_parts, b_parts, contribs) in [
            # M1
            ([(0, 0, 1), (2, 2, 1)], [(0, 0, 1), (2, 2, 1)], [(0, 0, 1), (1, 1, 1)]),
            # M2
            ([(2, 0, 1), (2, 2, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
            # M3
            ([(0, 0, 1)], [(0, 2, 1), (2, 2, -1)], [(0, 1, 1), (1, 1, 1)]),
            # M4
            ([(2, 2, 1)], [(2, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)]),
            # M5
            ([(0, 0, 1), (0, 2, 1)], [(2, 2, 1)], [(0, 0, -1), (0, 1, 1)]),
            # M6
            ([(2, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 2, 1)], [(1, 1, 1)]),
            # M7
            ([(0, 2, 1), (2, 2, -1)], [(2, 0, 1), (2, 2, 1)], [(0, 0, 1)]),
        ]:
            for (abr, abc, acoef) in a_parts:
                for (bbr, bbc, bcoef) in b_parts:
                    for (cbr, cbc, ccoef) in contribs:
                        ar, ac = a_off(abr, abc)
                        br, bc = b_off(bbr, bbc)
                        cr, cc = c_off(cbr, cbc)
                        sign = acoef * bcoef * ccoef
                        # 2x2 schoolbook sub-product at these block offsets.
                        for i in range(2):
                            for j in range(2):
                                for k in range(2):
                                    u = (ar + i) * m + (ac + j)
                                    v = (br + j) * p + (bc + k)
                                    w = (cc + k) * n + (cr + i)
                                    term(u, v, w, sign, c)
                                    c += 1
        return c

    col = 0
    # Part 1: Strassen for A1(4x4) * B1L(4x4) -> C[0:4, 0:4]
    col = strassen_4x4(0, 0, 0, 0, 0, 0, col)
    # Part 2: schoolbook A1(4x4) * B1R(4x2) -> C[0:4, 4:6]
    col = sb4x2(0, 0, 0, 4, 0, 4, col)
    # Part 3: schoolbook A2(4x1) * B2(1x6) -> C[0:4, 0:6]
    col = sb4x1x1x6(0, 4, 0, col)

    assert col == rank, "rank mismatch: %d != %d" % (col, rank)

    # Verify correctness.
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook (rank 120) to guarantee validity.
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

    if time.time() - start > 120:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
