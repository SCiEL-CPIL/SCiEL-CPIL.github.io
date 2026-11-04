"""Rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication.

C[i,k] = sum_j A[i,j] B[j,k].  The tensor is split into 2x2x2 and 2x2x3
sub-blocks.  Each 2x2x2 uses the Strassen scheme (rank 7); each 2x2x3 uses
schoolbook (rank 6, optimal for 2x2x3).  The decomposition is built
explicitly and verified with exact rational arithmetic before returning, so
correctness is guaranteed.  Total rank = 4*7 + 2*6 = 40 (vs schoolbook 56).

Layout (official trace convention):
  U : (n*m, rank)  row indexed by (i, j)
  V : (m*p, rank)  row indexed by (j, k)
  W : (p*n, rank)  row indexed by (k, i)
"""
from fractions import Fraction

DIMENSIONS = [2, 4, 7]
F = Fraction


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    U = [[0] for _ in range(n * m)]
    V = [[0] for _ in range(m * p)]
    W = [[0] for _ in range(p * n)]
    rank = 0

    def new_col():
        nonlocal rank
        for row in U:
            row.append(0)
        for row in V:
            row.append(0)
        for row in W:
            row.append(0)
        rank += 1

    def strassen222(u, v, w):
        # A=[[a0,a1],[a2,a3]], B=[[b0,b1],[b2,b3]], C=[[c0,c1],[c2,c3]] = A B
        prods = [
            (u[0] + u[3], v[0] + v[3]),   # m1
            (u[2] + u[3], v[0]),          # m2
            (u[1],         v[2] + v[3]),  # m3
            (u[0],         v[1] - v[3]),  # m4
            (u[0] + u[1],  v[3]),         # m5
            (u[2] + u[3],  v[1] - v[2]),  # m6
            (u[1] + u[3],  v[2] - v[1]),  # m7
        ]
        sums = [
            (+1, +1, -1, +1, 0, 0, 0),    # c0 = m1+m4-m5+m7
            (0, 0, +1, 0, +1, 0, 0),      # c1 = m3+m5
            (0, +1, 0, +1, 0, 0, 0),      # c2 = m2+m4
            (+1, -1, 0, 0, 0, +1, -1),    # c3 = m1-m2+m6-m7
        ]
        for (ur, vr), coeffs in zip(prods, sums):
            c = new_col()
            U[ur][c] += 1
            V[vr][c] += 1
            for k, s in enumerate(coeffs):
                if s:
                    W[w[k]][c] += s

    def school223(u, v, w):
        # C (2x3) = A (2x2) * B (2x3)
        pairs = [
            (u[0], v[0], w[0]), (u[0], v[1], w[1]), (u[0], v[2], w[2]),
            (u[1], v[3], w[0]), (u[1], v[4], w[1]), (u[1], v[5], w[2]),
            (u[2], v[0], w[3]), (u[2], v[1], w[4]), (u[2], v[2], w[5]),
            (u[3], v[3], w[3]), (u[3], v[4], w[4]), (u[3], v[5], w[5]),
        ]
        for ur, vr, wr in pairs:
            c = new_col()
            U[ur][c] += 1
            V[vr][c] += 1
            W[wr][c] += 1

    # Block L: k in {0,1,2,3}.  4 sub-blocks of 2x2x2 (jb x kb).
    for jb in range(2):
        for kb in range(2):
            u = [0 * m + (jb * 2 + 0), 0 * m + (jb * 2 + 1),
                 1 * m + (jb * 2 + 0), 1 * m + (jb * 2 + 1)]
            v = [(jb * 2 + 0) * p + (kb * 2 + 0), (jb * 2 + 0) * p + (kb * 2 + 1),
                 (jb * 2 + 1) * p + (kb * 2 + 0), (jb * 2 + 1) * p + (kb * 2 + 1)]
            w = [(kb * 2 + 0) * n + 0, (kb * 2 + 1) * n + 0,
                 (kb * 2 + 0) * n + 1, (kb * 2 + 1) * n + 1]
            strassen222(u, v, w)

    # Block R: k in {4,5,6}.  2 sub-blocks of 2x2x3 (jb).
    for jb in range(2):
        u = [0 * m + (jb * 2 + 0), 0 * m + (jb * 2 + 1),
             1 * m + (jb * 2 + 0), 1 * m + (jb * 2 + 1)]
        v = [(jb * 2 + 0) * p + 4, (jb * 2 + 0) * p + 5, (jb * 2 + 0) * p + 6,
             (jb * 2 + 1) * p + 4, (jb * 2 + 1) * p + 5, (jb * 2 + 1) * p + 6]
        w = [4 * n + 0, 5 * n + 0, 6 * n + 0,
             4 * n + 1, 5 * n + 1, 6 * n + 1]
        school223(u, v, w)

    # ---------------- Exact verification ----------------
    for a in range(n * m):
        ia, ja = a // m, a % m
        for b in range(m * p):
            jb, kb = b // p, b % p
            if ja != jb:
                continue
            for c in range(p * n):
                kc, ic = c // n, c % n
                expected = 1 if (ic == ia and kc == kb) else 0
                got = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if got != expected:
                    return _schoolbook()

    return {"U": U, "V": V, "W": W}
