"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 5]
_DEADLINE = 150.0
_T0 = time.time()


def _remaining():
    return _DEADLINE - 10.0 - (time.time() - _T0)  # 10s safety margin


# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = Fraction(1)
                V[j * p + k][c] = Fraction(1)
                W[k * n + i][c] = Fraction(1)
                c += 1
    return U, V, W


def _tensor(U, V, W, n, m, p):
    """Return the multiplication tensor T[(i,j,k)] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]."""
    rank = len(U[0])
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                T[(i, j, k)] = s
    return T


def _target_tensor(n, m, p):
    # T[(i,j,k)] = 1 if ... actually the multiplication tensor entry for (i,j,k)
    # is 1 (the product a_{i,j} * b_{j,k} contributes to c_{i,k}).
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i, j, k)] = Fraction(1)
    return T


def _verify(U, V, W, n, m, p):
    T = _tensor(U, V, W, n, m, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[(i, j, k)] != 1:
                    return False
    return True


def _try_reduce(U, V, W, n, m, p, max_iters=200000):
    """Try to remove a rank-1 column c if the tensor is preserved (linear dependency)."""
    rank = len(U[0])
    # Precompute target
    target = _target_tensor(n, m, p)
    # We'll greedily try to drop columns. For column c, check if there exist
    # integer/half-integer coefficients on the other columns reproducing the
    # contribution. Simplest exact test: if column c is exactly zero in one of
    # the three matrices, dropping it is safe. More generally, we attempt a
    # rational linear solve.
    changed = True
    while changed and _remaining() > 0:
        changed = False
        rank = len(U[0])
        for c in range(rank):
            if _remaining() <= 0:
                return U, V, W
            # Try to drop column c: need the remaining columns to still give target.
            # Exact test via solving for the "delta" tensor that column c provides:
            # delta[i,j,k] = U[i,j,c]*V[j,k,c]*W[k,i,c].
            # We need sum_{c' != c} U[i,j,c'] V[j,k,c'] W[k,i,c'] = target - delta.
            # This is a system in the (n*m + m*p + p*n) * (rank-1) variables —
            # generally intractable. Instead use a cheap sufficient condition:
            # if the column is a scalar multiple of another column in all three
            # matrices, merge.
            merged = False
            for d in range(rank):
                if d == c:
                    continue
                # check U[:,c] = alpha*U[:,d], V[:,c]=alpha*V[:,d], W[:,c]=alpha*W[:,d]
                alpha = None
                ok = True
                for idx in range(len(U)):
                    u1, u2 = U[idx][c], U[idx][d]
                    if u2 != 0:
                        a = u1 / u2
                        if alpha is None:
                            alpha = a
                        elif alpha != a:
                            ok = False
                            break
                    else:
                        if u1 != 0:
                            ok = False
                            break
                if not ok:
                    continue
                # verify with V and W
                ok = True
                for idx in range(len(V)):
                    v1, v2 = V[idx][c], V[idx][d]
                    if v2 != 0:
                        a = v1 / v2
                        if alpha is None:
                            alpha = a
                        elif alpha != a:
                            ok = False
                            break
                    else:
                        if v1 != 0:
                            ok = False
                            break
                if not ok:
                    continue
                ok = True
                for idx in range(len(W)):
                    w1, w2 = W[idx][c], W[idx][d]
                    if w2 != 0:
                        a = w1 / w2
                        if alpha is None:
                            alpha = a
                        elif alpha != a:
                            ok = False
                            break
                    else:
                        if w1 != 0:
                            ok = False
                            break
                if not ok:
                    continue
                # merge: U[:,d] += U[:,c], etc.
                for idx in range(len(U)):
                    U[idx][d] += U[idx][c]
                for idx in range(len(V)):
                    V[idx][d] += V[idx][c]
                for idx in range(len(W)):
                    W[idx][d] += W[idx][c]
                # drop column c
                for idx in range(len(U)):
                    U[idx].pop(c)
                for idx in range(len(V)):
                    V[idx].pop(c)
                for idx in range(len(W)):
                    W[idx].pop(c)
                merged = True
                changed = True
                break
            if merged:
                break
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    # Attempt rank reduction (cheap, exact, bounded).
    if _remaining() > 0:
        U, V, W = _try_reduce(U, V, W, n, m, p)
    # Final exact verification; fall back to schoolbook if anything is off.
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
    # Convert to plain numbers (int or half-int) for the interface.
    def conv(M):
        out = []
        for row in M:
            r = []
            for v in row:
                if v.denominator == 1:
                    r.append(v.numerator)
                else:
                    r.append(v.numerator / v.denominator)
            out.append(r)
        return out
    return {"U": conv(U), "V": conv(V), "W": conv(W)}
# EVOLVE-BLOCK-END
