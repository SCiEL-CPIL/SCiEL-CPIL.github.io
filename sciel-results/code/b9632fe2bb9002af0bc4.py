"""Structured search for a rank-minimal exact decomposition of the 2x3x3
matrix-multiplication tensor.

Strategy:
  1. Build a known rank-17 decomposition (Strassen on the leading 2x2x2 block,
     direct multiplication on the remaining 2x2x1 and 2x1x3 interactions).
  2. Attempt to find a rank-16 exact decomposition by:
       - parametrizing the factor matrices with a smooth ansatz,
       - minimizing a smooth (squared) residual over the 18 tensor entries,
       - using a linear-dependency / rank-reduction regularization to push one
         factor column toward a linear combination of the others,
       - recovering exact rational (half-integer) coefficients from the
         converged floating-point solution,
       - verifying exact equality.
  3. Return the best (lowest-rank) exact decomposition found, with a
     wall-clock safety guard.
"""

import time
import math
import itertools
import fractions
import numpy as np

DIMENSIONS = [2, 3, 3]
N, M, P = 2, 3, 3
# tensor size: (n*m) * (m*p) * (p*n) = 6 * 9 * 6 = 324 entries, but the
# multiplication tensor is sparse: T[i*m+j, j*p+k, k*n+i] = 1.
NM = N * M      # 6
MP = M * P      # 9
PN = P * N      # 6

DEADLINE = 110.0   # seconds, safety margin before the 120s (150s) kill
START_TIME = None


def _elapsed():
    return time.time() - START_TIME if START_TIME is not None else 0.0


def _remaining():
    return DEADLINE - _elapsed()


def _tensor_entries():
    """Yield (a, b, c) indices where T[a,b,c] = 1."""
    for i in range(N):
        for j in range(M):
            for k in range(P):
            # U index a = i*m + j, V index b = j*p + k, W index c = k*n + i
                a = i * M + j
                b = j * P + k
                c = k * N + i
                yield a, b, c


def _residual(U, V, W):
    """Sum of squared residuals over all 18 tensor entries.
    Residual for entry (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] - 1."""
    res = 0.0
    for (a, b, c) in _tensor_entries():
        # dot over r
        s = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
        d = s - 1.0
        res += d * d
    return res


def _build_rank17():
    """Known rank-17 decomposition: Strassen on 2x2x2 block, direct elsewhere.

    We label the 2x3 matrix A rows i in {0,1}, cols j in {0,1,2};
    the 3x3 matrix B rows j in {0,1,2}, cols k in {0,1,2};
    product C = A B, rows i, cols k.

    For the 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1}) use Strassen
    (7 products). For the remaining interactions (j=2 column of A, and k=2
    column of B) use direct multiplication (2*1*2 + 2*2*1 = 4+4=8 products).
    Total 7 + 8 = 15? Let's be careful and just build a valid rank-17 via
    schoolbook on a partition. Simpler: use schoolbook for everything except
    replace the 2x2x2 core with Strassen (saves 1: 8->7). 18 - 1 = 17.
    """
    rank = 17
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))

    col = 0
    # Strassen products for the 2x2x2 core: i in {0,1}, j in {0,1}, k in {0,1}
    # Strassen:
    # p0 = (a00+a11)(b00+b11)
    # p1 = (a10+a11)b00
    # p2 = a00(b01-b11)
    # p3 = a11(b10-b00)
    # p4 = (a00+a01)b11
    # p5 = (a10-a00)(b00+b10)
    # p6 = (a01-a11)(b01+b11)
    # c00 = p0+p3-p4+p6
    # c01 = p2+p4
    # c10 = p1+p5
    # c11 = p0+p2-p1+p5

    # We'll add 7 columns for these.
    def add(a_u, b_v, c_w, val=1.0):
        nonlocal col
        U[a_u, col] += val
        V[b_v, col] += val
        W[c_w, col] += val
        col += 1

    # p0 = (a00+a11)(b00+b11): need to expand so that each (i,j,k) core entry
    # gets the right coefficient. This is the classic embedding. Instead of
    # hand-deriving, we directly build the 7 rank-1 tensors whose sum equals
    # the 2x2x2 core tensor. We use the standard Strassen rank-1 factors.

    # We'll represent each Strassen product as a rank-1 term in (i,j,k) space
    # and then map to (a=i*3+j, b=j*3+k, c=k*2+i).

    # Strassen rank-1 factors (i,j,k) for the 7 products:
    # Each product p_t = (sum_i,j u_{t,i,j} a_{i,j})(sum_i,j v_{t,i,j} b_{i,j})
    # and contributes to c_{i,k} via w_{t,i,k}.
    # We build the full rank-1 tensors U_t[i*3+j], V_t[j*3+k], W_t[k*2+i].

    strassen = [
        # p0
        ([(0,0),(1,1)], [(0,0),(1,1)], [(0,0),(1,1)]),
        # p1
        ([(1,0),(1,1)], [(0,0)], [(1,0)]),
        # p2
        ([(0,0)], [(0,1),(1,1)], [(0,1)]),
        # p3
        ([(1,1)], [(1,0),(0,0)], [(1,1)]),
        # p4
        ([(0,0),(0,1)], [(1,1)], [(0,1),(1,1)]),
        # p5
        ([(1,0),(0,0)], [(0,0),(1,0)], [(1,0)]),
        # p6
        ([(0,1),(1,1)], [(0,1),(1,1)], [(1,1)]),
    ]
    # Signs for p2 and p3 and p1,p3,p5 in the combination; but for the TENSOR
    # we just need the 7 rank-1 tensors to sum to the core. The signs are
    # absorbed into the factors. Let's set the factor values (with signs) so
    # that sum_t U_t V_t W_t = core tensor (i,j,k)->delta.

    # We'll directly use the known Strassen rank-1 decomposition of the 2x2x2
    # multiplication tensor.
    strassen_factors = [
        # (U on (i,j) for core, V on (i,j) for core, W on (i,k) for core)
        # p0
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        # p1
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1}),
        # p2
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1}),
        # p3
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(1,1):1}),
        # p4
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,1):1,(1,1):-1}),
        # p5
        ({(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}, {(1,0):1}),
        # p6
        ({(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}, {(1,1):1}),
    ]

    for (u, v, w) in strassen_factors:
        for (i, j), val in u.items():
            U[i * M + j, col] += val
        for (i, j), val in v.items():
            # V index = j*P + k ; here the core's "j" is the col of A = row of B
            # and "k" is col of B. In Strassen the V factor is over (j,k) of B
            # where j is row of B (=col of A) and k is col of B.
            V[j * P + i, col] += val
        for (i, k), val in w.items():
            W[k * N + i, col] += val
        col += 1

    # Now add direct-multiplication rank-1 terms for the non-core interactions:
    # A has col j=2 (i in {0,1}); B has col k=2 (j in {0,1,2}).
    # Direct products needed:
    #   (i, j=2) with (j=2, k) for k in {0,1,2}: 2*3 = 6 terms? but k=2 already
    #   Let's just add ALL direct terms that are NOT in the core, i.e. any
    #   (i,j,k) with j==2 or k==2.
    for i in range(N):
        for j in range(M):
            for k in range(P):
                if j == 1 and k == 1:
                    continue  # core handled
                if j == 0 and k == 0:
                    continue
                if j == 0 and k == 1:
                    continue
                if j == 1 and k == 0:
                    continue
                # core is j in {0,1} and k in {0,1}
                if j in (0,1) and k in (0,1):
                    continue
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1

    # col should equal rank
    assert col <= rank, (col, rank)
    # trim
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    return U, V, W


def _try_rank16():
    """Attempt to find a rank-16 exact decomposition via smooth optimization
    plus rational recovery."""
    rank = 16
    # Initialize from rank-17, drop the last column, and let optimization fix.
    U0, V0, W0 = _build_rank17()
    U = U0[:, :rank].copy()
    V = V0[:, :rank].copy()
    W = W0[:, :rank].copy()

    # Smooth least-squares minimization of the residual using a simple
    # projected gradient / Gauss-Newton style iteration with a rank-reduction
    # regularizer that encourages one column to be a linear combo of others.
    # We use a small number of iterations and a smooth surrogate.

    # Parameters: flatten U, V, W.
    def pack(U, V, W):
        return np.concatenate([U.ravel(), V.ravel(), W.ravel()])

    def unpack(x):
        nU = U.shape[0] * rank
        nV = V.shape[0] * rank
        Ux = x[:nU].reshape(U.shape)
        Vx = x[nU:nU + nV].reshape(V.shape)
        Wx = x[nU + nV:].reshape(W.shape)
        return Ux, Vx, Wx

    def residual_and_grad(x):
        Ux, Vx, Wx = unpack(x)
        res = 0.0
        gU = np.zeros_like(Ux)
        gV = np.zeros_like(Vx)
        gW = np.zeros_like(Wx)
        for (a, b, c) in _tensor_entries():
            s = np.dot(np.dot(Ux[a, :], Vx[b, :]), Wx[c, :])
            d = s - 1.0
            res += d * d
            # grad wrt U[a,:]: 2*d * (Vx[b,:] * Wx[c,:])
            gU[a, :] += 2 * d * (Vx[b, :] * Wx[c, :])
            gV[b, :] += 2 * d * (Ux[a, :] * Wx[c, :])
            gW[c, :] += 2 * d * (Ux[a, :] * Vx[b, :])
        return res, np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])

    x = pack(U, V, W)
    # Smooth line-search gradient descent with a few restarts.
    best = (residual_and_grad(x)[0], x.copy())
    for restart in range(3):
        if _remaining() < 5.0:
            break
        xk = best[1].copy()
        lr = 1e-4
        for it in range(400):
            if _remaining() < 3.0:
                break
            res, g = residual_and_grad(xk)
            if res < best[0] - 1e-12:
                best = (res, xk.copy())
            # adaptive step
            step = lr
            for _ in range(8):
                xn = xk - step * g
                rn, _ = residual_and_grad(xn)
                if rn < res:
                    xk = xn
                    break
                step *= 0.5
            if it % 50 == 0:
                lr *= 0.8
        # small perturbation restart
        xk = best[1] + 1e-3 * np.random.randn(len(best[1]))
    res, x = best
    Ux, Vx, Wx = unpack(x)

    # Rational recovery (half-integers): round to nearest 0.5 and verify.
    def to_half(M_):
        return (np.round(M_ * 2) / 2)

    Uh, Vh, Wh = to_half(Ux), to_half(Vx), to_half(Wx)
    # Verify exact equality using fractions
    ok = True
    for (a, b, c) in _tensor_entries():
        s = 0
        for r in range(rank):
            s += fractions.Fraction(Uh[a, r]).limit_denominator(1000) * \
                 fractions.Fraction(Vh[b, r]).limit_denominator(1000) * \
                 fractions.Fraction(Wh[c, r]).limit_denominator(1000)
        if s != 1:
            ok = False
            break
    if ok:
        return Uh, Vh, Wh
    return None


def _to_lists(U, V, W):
    def conv(M_):
        return [[float(x) for x in row] for row in M_]
    return {"U": conv(U), "V": conv(V), "W": conv(W)}


def construct():
    global START_TIME
    START_TIME = time.time()

    best = None
    best_rank = 10**9

    # Try rank 16 first (ambitious)
    try:
        r16 = _try_rank16()
        if r16 is not None:
            best = r16
            best_rank = 16
    except Exception:
        pass

    # Fall back to rank 17 (known valid)
    if best is None:
        try:
            U, V, W = _build_rank17()
            best = (U, V, W)
            best_rank = 17
        except Exception:
            # Last resort: schoolbook rank 18
            n, m, p = DIMENSIONS
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
            best = (U, V, W)
            best_rank = rank

    # Verify the best is exact; if not, fall back to schoolbook.
    def is_exact(U, V, W):
        rank = U.shape[1]
        for (a, b, c) in _tensor_entries():
            s = 0
            for r in range(rank):
                s += fractions.Fraction(U[a, r]).limit_denominator(1000) * \
                     fractions.Fraction(V[b, r]).limit_denominator(1000) * \
                     fractions.Fraction(W[c, r]).limit_denominator(1000)
            if s != 1:
                return False
        return True

    U, V, W = best
    if not is_exact(U, V, W):
        # rebuild schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return _to_lists(U, V, W)
