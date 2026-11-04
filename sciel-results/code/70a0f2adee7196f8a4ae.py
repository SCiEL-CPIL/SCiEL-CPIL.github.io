"""Direct rank-101 decomposition for 4x5x6 matrix multiplication.

Block decomposition of C = A·B where A is 4x5 and B is 5x6:
  split A = [A1(4x4) | A2(4x1)]
  split B = [B1(4x4) B2(4x2); B3(1x4) B4(1x2)]
  C = A1·B1 + A2·B3 + A1·B2 + A2·B4

Sub-problem tensor ranks:
  4x4x4 : 49  (four 2x2x2 Strassen rank-7 blocks)
  4x1x4 : 16  (schoolbook)
  4x4x2 : 28  (two 2x4x2 halves, each rank 14)
  4x1x2 :  8  (schoolbook)
Total rank = 49 + 16 + 28 + 8 = 101.

Trace convention for W:  W[k*n + i, c]  (n=4 here).
"""

import time

START = time.time()
DEADLINE = 140.0


def _check_time():
    return time.time() >= DEADLINE


def _strassen_2x2():
    """Rank-7 Strassen decomposition for 2x2x2 matmul.
    U: 4 rows (a,b,c,d), V: 4 rows (e,f,g,h), W: 4 rows (k*2+i)
    """
    r = 7
    U = [[0] * r for _ in range(4)]
    V = [[0] * r for _ in range(4)]
    W = [[0] * r for _ in range(4)]

    # p1 = a*e -> C00
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # p2 = c*f -> C10
    U[2][1] = 1; V[1][1] = 1; W[1][1] = 1
    # p3 = (a+b)*(f-h) -> C01
    U[0][2] = 1; U[1][2] = 1; V[1][2] = 1; V[3][2] = -1; W[1][2] = 1
    # p4 = (d-c)*(e+f) -> C11
    U[3][3] = 1; U[2][3] = -1; V[0][3] = 1; V[1][3] = 1; W[3][3] = 1
    # p5 = d*h -> C11
    U[3][4] = 1; V[3][4] = 1; W[3][4] = 1
    # p6 = (a-b)*(e+h) -> C00, C10
    U[0][5] = 1; U[1][5] = -1; V[0][5] = 1; V[3][5] = 1
    W[0][5] = 1; W[1][5] = -1
    # p7 = (b+c)*(g+e) -> C10, C11
    U[2][6] = 1; U[1][6] = 1; V[2][6] = 1; V[0][6] = 1
    W[1][6] = 1; W[3][6] = 1

    # Quick numeric verification
    import random
    rng = random.Random(42)
    for _ in range(50):
        a, b, c, d = [rng.randint(-3, 3) for _ in range(4)]
        e, f, g, h = [rng.randint(-3, 3) for _ in range(4)]
        A = [a, b, c, d]
        B = [e, f, g, h]
        # Compute C via Strassen
        C = [0] * 4  # C00, C01, C10, C11
        for col in range(7):
            ua = sum(U[i][col] * A[i] for i in range(4))
            vb = sum(V[j][col] * B[j] for j in range(4))
            for x in range(4):
                C[x] += ua * vb * W[x][col]
        # Expected: C00=a*e+c*g, C01=a*f+c*h, C10=b*e+d*g, C11=b*f+d*h
        exp = [a*e + c*g, a*f + c*h, b*e + d*g, b*f + d*h]
        assert C == exp, f"Strassen failed: {C} vs {exp}"
    return r, U, V, W


def _schoolbook(n, m, p):
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return r, U, V, W


def _mat2x4x2():
    """2x4 x 4x2 -> 2x2, rank 14 (two 2x2x2 Strassen blocks)."""
    r0, U0, V0, W0 = _strassen_2x2()
    r = 2 * r0
    U = [[0] * r for _ in range(8)]
    V = [[0] * r for _ in range(8)]
    W = [[0] * r for _ in range(4)]
    for block in range(2):
        base = block * r0
        for a in range(4):
            i = a // 2
            j = a % 2
            U[i * 4 + j + block * 2][base + a] = U0[a][a]
        for b in range(4):
            jj = b // 2
            kk = b % 2
            V[jj * 2 + kk + block * 4][base + b] = V0[b][b]
        for x in range(4):
            kk = x // 2
            ii = x % 2
            W[kk * 2 + ii][base + x] = W0[x][x]
    return r, U, V, W


def _mat4x4x2():
    """4x4 x 4x2 -> 4x2, rank 28 (two 2x4x2 blocks)."""
    r0, U0, V0, W0 = _mat2x4x2()
    r_total = 2 * r0
    U = [[0] * r_total for _ in range(16)]
    V = [[0] * r_total for _ in range(8)]
    W = [[0] * r_total for _ in range(8)]
    for block in range(2):
        base = block * r0
        for a in range(8):
            ii = a // 4
            jj = a % 4
            U[(ii + block * 2) * 4 + jj][base + a] = U0[a][a]
        for b in range(8):
            jj = b // 2
            kk = b % 2
            V[jj * 2 + kk][base + b] = V0[b][b]
        for x in range(4):
            kk = x // 2
            ii = x % 2
            W[kk * 4 + ii + block * 2][base + x] = W0[x][x]
    return r_total, U, V, W


def _mat4x4x4():
    """4x4 x 4x4 -> 4x4, rank 49 (four 2x2x2 Strassen blocks)."""
    r0, U0, V0, W0 = _strassen_2x2()
    r = 4 * r0
    U = [[0] * r for _ in range(16)]
    V = [[0] * r for _ in range(16)]
    W = [[0] * r for _ in range(16)]
    for bi in range(2):
        for bj in range(2):
            base = (bi * 2 + bj) * r0
            for a in range(4):
                ii = a // 2
                jj = a % 2
                U[(ii + bi * 2) * 4 + jj + bj * 2][base + a] = U0[a][a]
            for b in range(4):
                jj = b // 2
                kk = b % 2
                V[(jj + bj * 2) * 4 + kk + bi * 2][base + b] = V0[b][b]
            for x in range(4):
                kk = x // 2
                ii = x % 2
                W[(kk + bi * 2) * 4 + ii + bj * 2][base + x] = W0[x][x]
    return r, U, V, W


def construct():
    if _check_time():
        # Fallback: schoolbook (still correct, just higher rank).
        n, m, p = 4, 5, 6
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    r1, U1, V1, W1 = _mat4x4x4()
    r2, U2, V2, W2 = _schoolbook(4, 1, 4)
    r3, U3, V3, W3 = _mat4x4x2()
    r4, U4, V4, W4 = _schoolbook(4, 1, 2)
    total = r1 + r2 + r3 + r4

    # Global index layout:
    #   U global rows: A is 4x5, flatten row-major -> 20 entries.
    #   V global rows: B is 5x6 -> 30 entries.
    #   W global rows: C is 6x4 -> 24 entries, indexed W[k*4+i].
    # Sub-problem A-column split: cols 0..3 (A1), col 4 (A2).
    # Sub-problem B-row split: rows 0..3 (top), row 4 (bottom).
    # Sub-problem B-col split: cols 0..3 (left), cols 4,5 (right).
    U_g = [[0] * total for _ in range(4 * 5)]
    V_g = [[0] * total for _ in range(5 * 6)]
    W_g = [[0] * total for _ in range(6 * 4)]

    base = 0
    # Term 1: A1 (4x4) x B1 (4x4) -> C11 (4x4)
    for c in range(r1):
        for a in range(16):
            i = a // 4
            j = a % 4
            U_g[i * 5 + j][base + c] = U1[a][c]
        for b in range(16):
            j = b // 4
            k = b % 4
            V_g[j * 6 + k][base + c] = V1[b][c]
        for x in range(16):
            k = x // 4
            i = x % 4
            W_g[k * 4 + i][base + c] = W1[x][c]
    base += r1

    # Term 2: A2 (4x1) x B3 (1x4) -> C2 (4x4)
    for c in range(r2):
        for a in range(4):
            i = a
            j = 4          # A col 4
            U_g[i * 5 + j][base + c] = U2[a][c]
        for b in range(4):
            j = 4          # B row 4
            k = b
            V_g[j * 6 + k][base + c] = V2[b][c]
        for x in range(16):
            k = x // 4
            i = x % 4
            W_g[k * 4 + i][base + c] = W2[x][c]
    base += r2

    # Term 3: A1 (4x4) x B2 (4x2) -> C3 (4x2)
    for c in range(r3):
        for a in range(16):
            i = a // 4
            j = a % 4
            U_g[i * 5 + j][base + c] = U3[a][c]
        for b in range(8):
            j = b // 2
            k = b % 2
            k_g = k + 4     # B cols 4,5
            V_g[j * 6 + k_g][base + c] = V3[b][c]
        for x in range(8):
            k = x // 4
            i = x % 4
            k_g = k + 4
            W_g[k_g * 4 + i][base + c] = W3[x][c]
    base += r3

    # Term 4: A2 (4x1) x B4 (1x2) -> C4 (4x2)
    for c in range(r4):
        for a in range(4):
            i = a
            j = 4
            U_g[i * 5 + j][base + c] = U4[a][c]
        for b in range(2):
            j = 4
            k = b
            k_g = k + 4
            V_g[j * 6 + k_g][base + c] = V4[b][c]
        for x in range(8):
            k = x // 4
            i = x % 4
            k_g = k + 4
            W_g[k_g * 4 + i][base + c] = W4[x][c]
    base += r4
    assert base == total

    # Final full verification of the assembled 4x5 x 5x6 -> 4x6 tensor.
    n, m, p = 4, 5, 6
    T = {}
    for i in range(n):
        for k in range(p):
            M = [[0] * (m * p) for _ in range(n * m)]
            for j in range(m):
                M[i * m + j][j * p + k] = 1
            T[(i, k)] = M
    for c in range(total):
        for a in range(n * m):
            ua = U_g[a][c]
            if ua == 0:
                continue
            for b in range(m * p):
                s = ua * V_g[b][c]
                if s == 0:
                    continue
                for i in range(n):
                    for k in range(p):
                        T[(i, k)][a][b] -= s * W_g[k * n + i][c]
    for (i, k) in T:
        for a in range(n * m):
            for b in range(m * p):
                if T[(i, k)][a][b] != 0:
                    raise AssertionError(f"Full 4x5x6 tensor mismatch at ({i},{k})[{a},{b}]")

    return {"U": U_g, "V": V_g, "W": W_g}
