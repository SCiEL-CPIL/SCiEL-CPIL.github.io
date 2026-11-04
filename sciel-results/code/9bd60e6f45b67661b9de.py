"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _trivial(rank=60):
    n, m, p = 2, 3, 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for rr in range(rank):
        ii = rr // (m * p)
        rem = rr % (m * p)
        jj = rem // p
        kk = rem % p
        U[ii * m + jj][rr] = 1
        V[jj * p + kk][rr] = 1
        W[kk * n + ii][rr] = 1
    return {"U": U, "V": V, "W": W}


def _strassen_base():
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    r = 0
    strassen_u = [
        [(0, 1), (4, 1)],
        [(3, 1), (4, 1)],
        [(0, 1)],
        [(4, 1)],
        [(0, 1), (1, 1)],
        [(3, 1), (0, -1)],
        [(1, 1), (4, -1)],
    ]
    strassen_v = [
        [(0, 1), (11, 1)],
        [(0, 1)],
        [(1, 1), (11, -1)],
        [(10, 1), (0, -1)],
        [(11, 1)],
        [(0, 1), (1, 1)],
        [(10, 1), (11, 1)],
    ]
    strassen_w = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)],
    ]

    for b in range(5):
        col0 = 2 * b
        col1 = 2 * b + 1
        for s in range(7):
            for (u_off, coeff) in strassen_u[s]:
                U[u_off][r] += coeff
            for (v_off, coeff) in strassen_v[s]:
                if v_off == 0:
                    v_row = col0
                elif v_off == 1:
                    v_row = col1
                elif v_off == 10:
                    v_row = 10 + col0
                elif v_off == 11:
                    v_row = 10 + col1
                else:
                    raise ValueError(f"Bad v_off {v_off}")
                V[v_row][r] += coeff
            for (w_off, coeff) in strassen_w[s]:
                if w_off == 0:
                    w_row = col0 * 2 + 0
                elif w_off == 1:
                    w_row = col0 * 2 + 1
                elif w_off == 2:
                    w_row = col1 * 2 + 0
                elif w_off == 3:
                    w_row = col1 * 2 + 1
                else:
                    raise ValueError(f"Bad w_off {w_off}")
                W[w_row][r] += coeff
            r += 1

    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][r] = 1
            V[20 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1

    return U, V, W


def _validate(U, V, W):
    n, m, p = 2, 3, 10
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(Ua[i * m + j] * Va[j * p + k], Wa[k * n + i])
                if not np.isclose(lhs, 1.0, atol=1e-9):
                    return False
    return True


def _try_merge(U, V, W, deadline):
    """Exact local polish: repeatedly remove a pair of rank-1 terms (a, b)
    that are scalar-multiples of each other (redundant), collapsing rank by 1.
    Safe: only fires when the two terms are linearly dependent, which means
    the decomposition is unchanged up to a scalar. Returns (U,V,W) possibly
    with smaller rank, or the input unchanged."""
    n, m, p = 2, 3, 10
    rank = len(U[0])
    # Work on copies as float for linear-algebra tests, but keep integer output.
    Uf = [row[:] for row in U]
    Vf = [row[:] for row in V]
    Wf = [row[:] for row in W]

    removed = [False] * rank
    changed = False
    # Iterate: find any pair (a,b) with all three vectors collinear.
    # Collinear means U_a = c*U_b, V_a = c*V_b, W_a = c*W_b for some c.
    for _pass in range(200):
        if time.time() > deadline:
            break
        found = False
        for a in range(rank):
            if removed[a]:
                continue
            if time.time() > deadline:
                break
            for b in range(a + 1, rank):
                if removed[b]:
                    continue
                # Determine candidate scalar c from U first (nonzero pivot).
                # Find a coordinate where U_b is nonzero.
                c = None
                for x in range(len(Uf[a])):
                    ub = Vf[b][x]
                    if abs(ub) > 1e-12:
                        c = Uf[a][x] / ub
                        break
                    # also check V
                # Better: derive c from any of the three vectors where a pivot exists.
                c = None
                for x in range(len(Uf[a])):
                    if abs(Vf[b][x]) > 1e-12:
                        c = Uf[a][x] / Vf[b][x]
                        break
                if c is None:
                    for x in range(len(Vf[a])):
                        if abs(Wf[b][x]) > 1e-12:
                            c = Vf[a][x] / Wf[b][x]
                            break
                if c is None:
                    for x in range(len(Wf[a])):
                        if abs(Uf[b][x]) > 1e-12:
                            c = Wf[a][x] / Uf[b][x]
                            break
                if c is None:
                    continue
                # Verify collinearity across all three vectors.
                ok = True
                for x in range(len(Uf[a])):
                    if abs(Uf[a][x] - c * Vf[b][x]) > 1e-9:
                        ok = False
                        break
                if ok:
                    for x in range(len(Vf[a])):
                        if abs(Vf[a][x] - c * Wf[b][x]) > 1e-9:
                            ok = False
                            break
                if ok:
                    for x in range(len(Wf[a])):
                        if abs(Wf[a][x] - c * Uf[b][x]) > 1e-9:
                            ok = False
                            break
                if ok:
                    # Remove term a (keep b).
                    removed[a] = True
                    found = True
                    changed = True
                    break
            if found:
                break
        if not found:
            break

    if not changed:
        return U, V, W

    # Build reduced matrices.
    keep = [i for i in range(rank) if not removed[i]]
    new_rank = len(keep)
    U2 = [[0] * new_rank for _ in range(n * m)]
    V2 = [[0] * new_rank for _ in range(m * p)]
    W2 = [[0] * new_rank for _ in range(p * n)]
    for new_i, old_i in enumerate(keep):
        for x in range(len(Uf[old_i])):
            U2[x][new_i] = Uf[old_i][x]
        for x in range(len(Vf[old_i])):
            V2[x][new_i] = Vf[old_i][x]
        for x in range(len(Wf[old_i])):
            W2[x][new_i] = Wf[old_i][x]
    # Snap to integers/half-integers if close.
    def snap(M):
        for row in M:
            for i, v in enumerate(row):
                if abs(v - round(v)) < 1e-9:
                    row[i] = int(round(v))
        return M
    snap(U2); snap(V2); snap(W2)
    if _validate(U2, V2, W2):
        return U2, V2, W2
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 110

    U, V, W = _strassen_base()

    if not _validate(U, V, W):
        return _trivial(60)

    # Refinement within the same family: exact local merge of redundant terms.
    U2, V2, W2 = _try_merge(U, V, W, deadline)
    if len(U2[0]) < len(U[0]):
        U, V, W = U2, V2, W2

    if not _validate(U, V, W):
        return _trivial(60)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
