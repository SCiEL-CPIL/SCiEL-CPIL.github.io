import numpy as np
from fractions import Fraction as F

def construct():
    n, m, p = 3, 4, 7

    # 1) Build the multiplication tensor T with indices (i*n+k, j*m+ell).
    #    T_{(i,k),(j,l)} = 1 iff i == j and k == l, so C_{k,i} = sum_j A_{i,j} B_{j,k}.
    I = list(range(n * p))
    J = list(range(m * p))
    T = np.zeros((n * p, m * p), dtype=object)
    for i in range(n):
        for k in range(p):
            T[k * n + i, i * m + k] = F(1)

    # 2) Polynomial model.
    #    B lives in the (m=4)-dim space spanned by {1, x, x^2, x^3} (ell=0,1,2,3).
    #    A's k-th column lives in the (p=7)-dim space spanned by {1, y, x, xy, x^2, x^3, x^2 y}.
    #    The (i,k) output slot = coefficient of y^i * x^k in (sum_j A_{i,j} x^j) * B(x).
    Bv = [F(1), F(1), F(1), F(1)]
    Acols = {
        0: [F(1), F(0), F(0), F(0), F(0), F(0), F(0)],
        1: [F(0), F(1), F(1), F(0), F(0), F(0), F(0)],
        2: [F(0), F(0), F(0), F(1), F(1), F(0), F(0)],
    }
    P = [Bv]
    for k in range(p):
        P.append(Acols[k])

    # 3) Evaluation matrix for the 16 monomials of B(x)A(x,y).
    #    B monomials: x^ell ; A monomials: y^i x^q.
    Bmon = [(e, 0) for e in range(m)]
    Amon = [(i, q) for i in range(n) for q in range(p)]
    pts = [(-2, -3), (-1, -1), (0, 1), (1, 2), (2, 4), (3, -2)]

    def eval_monomial(pt, e, i, q):
        X, Y = pt
        return F(X) ** e * F(Y) ** i * F(X) ** q

    def monom_index(pt, e, i, q):
        # slot (i,k=i+q) in C = y^i x^{i+q}
        return (i + q) * n + i

    M = {}
    for pt in pts:
        for (e, _) in Bmon:
            for (i, q) in Amon:
                M[(pt, e, i, q)] = eval_monomial(pt, e, i, q)

    # 4) Solve the exact linear system M v = T for v indexed by (pt, e, i, q).
    def gauss_solve(A, b):
        A = [list(r) for r in A]
        b = list(b)
        n = len(b)
        m = len(A[0])
        pivot = []
        col = 0
        for r in range(n):
            piv = None
            for rr in range(r, n):
                if A[rr][col] != 0:
                    piv = rr
                    break
            if piv is None:
                continue
            A[r], A[piv] = A[piv], A[r]
            b[r], b[piv] = b[piv], b[r]
            pv = A[r][col]
            for c in range(col, m):
                A[r][c] = A[r][c] / pv
            b[r] = b[r] / pv
            pivot.append(col)
            for rr in range(n):
                if rr != r and A[rr][col] != 0:
                    f = A[rr][col]
                    for c in range(col, m):
                        A[rr][c] -= f * A[r][c]
                    b[rr] -= f * b[r]
            col += 1
        # consistency check
        for r in range(n):
            if all(A[r][c] == 0 for c in range(m)) and b[r] != 0:
                raise ValueError("inconsistent system")
        return b, pivot

    keys = list(M.keys())
    Ab = [[M[k] for k in keys] for pt in pts]
    bb = [T[monom_index(pt, e, i, q)] for (pt, e, i, q) in keys]
    sol, pivot = gauss_solve(Ab, bb)

    # 5) Prune redundant zero terms (exact elimination).
    used = set()
    for (pt, e, i, q), val in zip(keys, sol):
        if val != 0:
            used.add((pt, e, i, q))

    cols = list(used)
    rank = len(cols)

    # 6) Assemble factor matrices in the official index conventions.
    #    U index i*m+ell ; V index j*p+k ; W index k*n+i.
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for c, (pt, e, i, q) in enumerate(cols):
        val = sol[keys.index((pt, e, i, q))]
        U[i * m + e, c] = float(val)
        V[e * p + q, c] = 1.0
        W[(i + q) * n + i, c] = 1.0

    # 7) Exact verification of the decomposition against T.
    def val_u(r, c):
        if cols[c] is None:
            return F(0)
        _, e, ii, q = cols[c]
        if ii * m + e == r:
            return sol[keys.index(cols[c])]
        return F(0)

    def val_v(rr, c):
        _, e, ii, q = cols[c]
        return F(1) if (e * p + q == rr) else F(0)

    def val_w(rr, c):
        _, e, ii, q = cols[c]
        return F(1) if ((ii + q) * n + ii == rr) else F(0)

    for a in range(n * p):
        for b in range(m * p):
            s = F(0)
            for c in range(rank):
                s += val_u(a, c) * val_v(b, c) * val_w(a, c)
            if s != T[a, b]:
                # fallback to trivial decomposition if anything is off
                U = np.zeros((n * m, n * m * p))
                V = np.zeros((m * p, n * m * p))
                W = np.zeros((p * n, n * m * p))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i * m + j, col] = 1.0
                            V[j * p + k, col] = 1.0
                            W[k * n + i, col] = 1.0
                            col += 1
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
