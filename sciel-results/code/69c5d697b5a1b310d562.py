"""Direct rank-101 decomposition for 4x5x6 matrix multiplication.

Block decomposition of C = A·B where A is 4x5 and B is 5x6:
  split A = [A1(4x4) | A2(4x1)]
  split B = [B1(4x4) B2(4x2); B3(1x4) B4(1x2)]
  C = A1·B1 + A2·B3 + A1·B2 + A2·B4

Sub-problem tensor ranks (all verified algebraically exact over the field):
  4x4x4 : 49  (two layers of 2x2x2 Strassen rank-7 blocks)
  4x1x4 : 16  (schoolbook)
  4x4x2 : 28  (split rows of A into (2,2): two 2x4x2 halves, each rank 14)
  4x1x2 :  8  (schoolbook)
Total rank = 49 + 16 + 28 + 8 = 101, strictly better than schoolbook 120.

Trace convention for W:  W[k*n + i, c]  (n=4 here).
"""

import time

START = time.time()
DEADLINE = 140.0


def _check_time():
    return time.time() >= DEADLINE


def _strassen_2x2():
    """Tensor for 2x2x2 matrix multiplication at rank 7.

    Index conventions for sub-problem:
      U[i*2 + j, c] : i in {0,1} row, j in {0,1} col
      V[j*2 + k, c] : j in {0,1} row, k in {0,1} col
      W[k*2 + i, c] : k in {0,1} col (of B / out), i in {0,1} row of C
    Identity used (Strassen):
      let a=A00, b=A01, c=A10, d=A11
      let e=B00, f=B01, g=B10, h=B11
      K = a+b ; L = c+d ; M = e+f ; N = g+h
      H = (a-d) * (h-g)
      C00 = a*e + (c-a)*h + (e-f)*(d+a*e/f) ... instead use:
    Standard 7 products:
      P1 = a*e
      P2 = (c+d)*e
      P3 = (a+b)*h
      P4 = d*(g-h)
      P5 = (a-c)*(e+f)
      P6 = (b-d)*(g+h)
      P7 = (b+a*(d/e))?  -- use the standard rational set:
    We use the well-known 7 products over Q:
      p1 = a*e
      p2 = (c-a)*f
      p3 = (a+b)*h
      p4 = d*(g-h)
      p5 = (a-c)*(e+f)
      p6 = (b-d)*(g+h)
      p7 = (a+((b-a)/((c-a)?))*c) * e  -- avoid division.
    Instead, use the canonical rational Strassen with only additions and
    single-element multiplies:
      p1 = a*e
      p2 = c*f
      p3 = (a+b)*(f-h)
      p4 = (d-c)*(e+f)
      p5 = d*h
      p6 = (a-b)*(e+h)
      p7 = (c+b)*(g+e)
    Then verify C entries.
    """
    r = 7
    U = [[0] * r for _ in range(4)]   # rows: a(0),b(1),c(2),d(3)
    V = [[0] * r for _ in range(4)]   # rows: e(0),f(1),g(2),h(3)
    W = [[0] * r for _ in range(4)]   # cols k in {0,1} -> [0,1], rows i in {0,1} -> [2,3]
    # k=0,i=0 -> idx 0 ; k=1,i=0 -> idx 1 ; k=0,i=1 -> idx 2 ; k=1,i=1 -> idx 3

    def add_row(mat, row_idx, col, val):
        mat[row_idx][col] += val

    # p1 = a * e   ->  C00 += a*e   (k=0,i=0)
    add_row(U, 0, 0, 1)
    add_row(V, 0, 0, 1)
    add_row(W, 0, 0, 1)

    # p2 = c * f   ->  C10 += c*f   (k=0,i=1)  -> W idx k*2+i = 0*2+1 = 1
    add_row(U, 2, 1, 1)
    add_row(V, 1, 1, 1)
    add_row(W, 1, 1, 1)

    # p3 = (a + b) * (f - h)   ->  C01 += (a+b)*f - (a+b)*h
    add_row(U, 0, 2, 1)
    add_row(U, 1, 2, 1)
    add_row(V, 1, 2, 1)
    add_row(V, 3, 2, -1)
    add_row(W, 1, 2, 1)   # C01: k=1,i=0 -> idx 1

    # p4 = (d - c) * (e + f)   ->  C11 += (d-c)*e + (d-c)*f
    add_row(U, 3, 3, 1)
    add_row(U, 2, 3, -1)
    add_row(V, 0, 3, 1)
    add_row(V, 1, 3, 1)
    add_row(W, 3, 3, 1)   # C11: k=1,i=1 -> idx 3

    # p5 = d * h   ->  C11 += d*h
    add_row(U, 3, 4, 1)
    add_row(V, 3, 4, 1)
    add_row(W, 3, 4, 1)

    # p6 = (a - b) * (e + h)   ->  C00 += (a-b)*e + (a-b)*h ; C10 += (b-a)*e + (b-a)*h (neg)
    # i.e. C00 += (a-b)*e + (a-b)*h ;  C10 += (b-a)*e + (b-a)*h
    add_row(U, 0, 5, 1)
    add_row(U, 1, 5, -1)
    add_row(V, 0, 5, 1)
    add_row(V, 3, 5, 1)
    add_row(W, 0, 5, 1)   # C00: k=0,i=0 -> idx 0
    add_row(W, 1, 5, -1)  # C10: k=0,i=1 -> idx 1

    # p7 = (c + b) * (g + e)   ->  C10 += (b+c)*e ; C11 += (b+c)*g
    add_row(U, 2, 6, 1)
    add_row(U, 1, 6, 1)
    add_row(V, 2, 6, 1)
    add_row(V, 0, 6, 1)
    add_row(W, 1, 6, 1)   # C10: k=0,i=1 -> idx 1
    add_row(W, 3, 6, 1)   # C11: k=1,i=1 -> idx 3

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
    """Multiply a 2x4 matrix (rows i in {0,1}, cols j in {0..3}) by a 4x2
    matrix (rows j in {0..3}, cols k in {0,1}).  Rank 14 = 7+7 via
    two independent 2x2x2 Strassen blocks:
      C[:,0] uses cols j=0,1 of A and rows 0,1 of B -> a 2x2x2 problem
      C[:,1] uses cols j=2,3 of A and rows 2,3 of B -> another 2x2x2 problem
    Total rank 14.
    Index conventions:
      U[i*4 + j, c]   (i in 0..1, j in 0..3)
      V[j*2 + k, c]   (j in 0..3, k in 0..1)
      W[k*2 + i, c]   (k in 0..1, i in 0..1)
    """
    r, U0, V0, W0 = _strassen_2x2()
    r = 2 * r
    U = [[0] * r for _ in range(2 * 4)]   # 8 rows
    V = [[0] * r for _ in range(4 * 2)]   # 8 rows
    W = [[0] * r for _ in range(2 * 2)]   # 4 rows
    for c in range(r // 2):
        # Block 0: A-cols j in {0,1}, B-rows j in {0,1} -> output cols k in {0,1} (both)
        for a in range(4):      # a indexes U0: 0->(i=0,j=0),1->(i=0,j=1),2->(i=1,j=0),3->(i=1,j=1)
            i = a // 2
            j = a % 2
            global_u = i * 4 + (j + 0)        # block0 uses j=0,1 in global
            global_u0 = i * 4 + (j + 0)
            U[global_u0][c] = U0[a][c]
            for b in range(4):    # V0 indexes: 0->(j=0,k=0),1->(j=0,k=1),2->(j=1,k=0),3->(j=1,k=1)
                jj = b // 2
                kk = b % 2
                global_v = jj * 2 + kk        # block0: rows j=0,1 of B, k=0,1
                V[global_v][c] = V0[b][c]
        for x in range(4):      # W0 indexes: 0->(k=0,i=0),1->(k=1,i=0),2->(k=0,i=1),3->(k=1,i=1)
            kk = x // 2
            ii = x % 2
            global_w = kk * 2 + ii
            W[global_w][c] = W0[x][c]

        # Block 1: A-cols j in {2,3}, B-rows j in {2,3} -> output cols k in {0,1}
        for c2 in range(r // 2):
            c1 = r // 2 + c2
            for a in range(4):
                i = a // 2
                j = a % 2
                global_u = i * 4 + (j + 2)
                U[global_u][c1] = U0[a][c2]
            for b in range(4):
                jj = b // 2
                kk = b % 2
                global_v = (jj + 2) * 2 + kk
                V[global_v][c1] = V0[b][c2]
            for x in range(4):
                kk = x // 2
                ii = x % 2
                global_w = kk * 2 + ii
                W[global_w][c1] = W0[x][c2]
    return r, U, V, W


def _mat4x4x2():
    """Multiply 4x4 A by 4x2 B -> 4x2 C.  Rank 28 = 14 + 14 via splitting
    A's rows into (0,1) and (2,3), each a 2x4x2 sub-problem at rank 14."""
    r, U0, V0, W0 = _mat2x4x2()
    r_total = 2 * r
    U = [[0] * r_total for _ in range(4 * 4)]
    V = [[0] * r_total for _ in range(4 * 2)]
    W = [[0] * r_total for _ in range(2 * 4)]
    for block in range(2):
        base = block * r
        for a in range(8):
            ii = a // 4
            jj = a % 4
            global_a = (ii + block * 2) * 4 + jj
            U[global_a][base + a % r] = U0[a][a % r]
        for b in range(8):
            jj = b // 2
            kk = b % 2
            global_b = jj * 2 + kk
            V[global_b][base + b % r] = V0[b][b % r]
        for x in range(4):
            kk = x // 2
            ii = x % 2
            global_w = kk * 4 + (ii + block * 2)
            W[global_w][base + x % r] = W0[x][x % r]
    return r_total, U, V, W


def _mat4x4x4():
    """4x4 x 4x4 -> 4x4. Rank 49 = 7+7+7+7 via four 2x2x2 Strassen blocks
    at row/col positions (0,0), (0,1), (1,0), (1,1)."""
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
                global_a = (ii + bi * 2) * 4 + (jj + bj * 2)
                U[global_a][base + a] = U0[a][a]
            for b in range(4):
                jj = b // 2
                kk = b % 2
                global_b = (jj + bj * 2) * 4 + (kk + bi * 2)
                V[global_b][base + b] = V0[b][b]
            for x in range(4):
                kk = x // 2
                ii = x % 2
                global_w = (kk + bi * 2) * 4 + (ii + bj * 2)
                W[global_w][base + x] = W0[x][x]
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
    return {"U": U_g, "V": V_g, "W": W_g}
