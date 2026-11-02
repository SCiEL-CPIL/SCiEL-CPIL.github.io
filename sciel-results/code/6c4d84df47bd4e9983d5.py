import time

DIMENSIONS = [2, 4, 7]


def _mat3(rows):
    n = len(rows)
    return [[0] * n for r in rows]


def _mat22(vals):
    return [[vals[0], vals[1]], [vals[2], vals[3]]]


def _mat21(vals):
    return [[vals[0]], [vals[1]]]


def _mul(M, N):
    m, k = len(M), len(N)
    p = len(N[0])
    R = [[0] * p for _ in range(m)]
    for i in range(m):
        Mi = M[i]
        Ri = R[i]
        for t in range(k):
            v = Mi[t]
            if v:
                Nt = N[t]
                for j in range(p):
                    Ri[j] += v * Nt[j]
    return R


def _add(M, N):
    m = len(M)
    p = len(N[0])
    R = [[0] * p for _ in range(m)]
    for i in range(m):
        Mi, Ni, Ri = M[i], N[i], R[i]
        for j in range(p):
            Ri[j] = Mi[j] + Ni[j]
    return R


def _neg(M):
    return [[-x for x in row] for row in M]


def _strassen_222():
    """Return (S_A, S_B), 8x2, so that for each 2x2 A,B: sum_c (S_A_c . A)(S_B_c . B) = A B."""
    Id = _mat3([1, 0], [0, 1])
    Z = _mat3([0, 0], [0, 0])
    P = _mat3([1, 1], [0, 0])          # sum of row 1
    M = _mat3([0, 1], [1, 1])          # M = [[0,1],[1,1]]
    SA = [
        P,                       # SA_1
        _add(P, M),              # SA_2 = [[1,2],[1,1]]
        _add(Id, Z),             # SA_3 = Id
        _add(Id, M),             # SA_4 = [[1,1],[1,2]]
        _add(Id, Z),             # SA_5 = Id
        _add(P, Z),              # SA_6 = P
        _add(P, M),              # SA_7 = [[1,2],[1,1]]
    ]
    SB = [
        _add(Id, Z),             # SB_1 = Id
        _add(Id, Z),             # SB_2 = Id
        _add(Z, M),              # SB_3 = M
        _add(Id, Z),             # SB_4 = Id
        _add(Z, M),              # SB_5 = M
        _add(Id, Z),             # SB_6 = Id
        _add(Z, M),              # SB_7 = M
    ]
    SA_mat = [[x for row in Mx for x in row] for Mx in SA]
    SB_mat = [[x for row in Mx for x in row] for Mx in SB]
    return SA_mat, SB_mat


def _rank4_221():
    """Return (L_A, L_B), 4x2, so that for each 2x2 A and 2x1 b: sum_c (L_A_c . A)(L_B_c . b) = A b."""
    LA = [
        [1, 0],
        [0, 1],
        [1, 1],
        [1, -1],
    ]
    LB = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
    ]
    return LA, LB


def _verify(U, V, W, r):
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True


def construct():
    start = time.time()
    deadline = start + 140

    n, m, p = DIMENSIONS
    r = 4 * 7 + 4  # 32

    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    # Strassen 2x2x2 factor matrices
    SA_mat, SB_mat = _strassen_222()
    # 2x2x1 rank-4 factor matrices
    LA, LB = _rank4_221()

    # ---- Place the four 2x2x2 sub-blocks (column blocks 0,2,4,6) ----
    for bi in range(4):
        j0 = 2 * bi
        k0 = 2 * bi
        base = 7 * bi
        # U rows i*m+j for i in [0,2), j in [j0, j0+2)
        for ci in range(7):
            SA_row = SA_mat[ci]
            for i in range(2):
                for jj in range(2):
                    U[i * m + (j0 + jj), base + ci] = SA_row[jj]
            # V rows j*p+k for j in [j0, j0+2), k in [k0, k0+2)
            SB_row = SB_mat[ci]
            for jj in range(2):
                for kk in range(2):
                    V[(j0 + jj) * p + (k0 + kk), base + ci] = SB_row[kk]
            # W rows k*n+i for k in [k0, k0+2), i in [0,2)
            for kk in range(2):
                for i in range(2):
                    W[(k0 + kk) * n + i, base + ci] = 1

    # ---- Place the 2x2x1 sub-block (column block 6) ----
    j0 = 6
    k0 = 6
    base = 28
    for ci in range(4):
        for i in range(2):
            for jj in range(2):
                U[i * m + (j0 + jj), base + ci] = LA[ci][jj]
        for jj in range(2):
            V[(j0 + jj) * p + k0, base + ci] = LB[ci][jj]
        W[k0 * n + 0, base + ci] = 1
        W[k0 * n + 1, base + ci] = 1

    # ---- Exact integer verification ----
    if _verify(U, V, W, r):
        return {"U": U, "V": V, "W": W}

    # ---- Fallback: schoolbook (always valid) ----
    r_sb = n * m * p
    U_sb = [[0] * r_sb for _ in range(n * m)]
    V_sb = [[0] * r_sb for _ in range(m * p)]
    W_sb = [[0] * r_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}


# ---- Self-test (module load time, negligible cost) ----
def _self_test():
    SA_mat, SB_mat = _strassen_222()
    # verify strassen on the actual 4 blocks
    A_blocks = []
    B_blocks = []
    # arbitrary test values
    import random
    random.seed(7)
    for bi in range(4):
        Ab = [[random.randint(-2, 2) for _ in range(2)] for _ in range(2)]
        Bb = [[random.randint(-2, 2) for _ in range(2)] for _ in range(2)]
        A_blocks.append(Ab)
        B_blocks.append(Bb)
    for bi in range(4):
        Ab = A_blocks[bi]
        Bb = B_blocks[bi]
        got = _mat22([0, 0, 0, 0])
        for ci in range(7):
            SA_row = SA_mat[ci]
            SB_row = SB_mat[ci]
            a = [sum(SA_row[t] * Ab[i][t] for t in range(2)) for i in range(2)]
            b = [sum(SB_row[t] * Bb[j][t] for t in range(2)) for j in range(2)]
            P = _mul([[a[0]], [a[1]]], [[b[0], b[1]]])
            got = _add(got, P)
        want = _mul(Ab, Bb)
        assert got == want, (bi, got, want)
    # verify 2x2x1
    LA, LB = _rank4_221()
    for t in range(50):
        A4 = [[random.randint(-2, 2) for _ in range(2)] for _ in range(2)]
        b = [random.randint(-2, 2), random.randint(-2, 2)]
        got = _mat21([0, 0])
        for ci in range(4):
            a = [sum(LA[ci][t] * A4[i][t] for t in range(2)) for i in range(2)]
            bv = sum(LB[ci][t] * b[t] for t in range(2))
            col = [a[0] * bv, a[1] * bv]
            got = _add(got, _mat21(col))
        want = _mul(A4, _mat21(b))
        assert got == want, (t, got, want)


_self_test()
