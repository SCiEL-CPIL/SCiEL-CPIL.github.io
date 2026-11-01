"""Combined program: Strassen-block structure + exact verification + new
deterministic subspace/rank-reduction search (exact rational reconstruction)."""
DIMENSIONS = [2, 2, 11]

import time
import random
import numpy as np
from fractions import Fraction

START = time.time()
DEADLINE = START + 118  # safety margin before the 150 s kill


def _time_left():
    return DEADLINE - time.time()


# ---------------------------------------------------------------------------
# Tensor bookkeeping (trace convention W[k*n+i, r])
# ---------------------------------------------------------------------------
N, M, P = 2, 2, 11
NM, MP, PN = N * M, M * P, P * N  # 4, 22, 22


def _tensor_entries():
    entries = []
    for i in range(N):
        for j in range(M):
            for k in range(P):
                entries.append((i * M + j, j * P + k, k * N + i))
    return entries


ENTRIES = _tensor_entries()
NSET = set(ENTRIES)


def _verify(U, V, W):
    """Exact verification: non-zero entries sum to 1, all others to 0."""
    rank = len(U[0])
    for (u, v, w) in ENTRIES:
        s = 0
        for r in range(rank):
            s += U[u][r] * V[v][r] * W[w][r]
        if s != 1:
            return False
    for u in range(NM):
        for v in range(MP):
            for w in range(PN):
                if (u, v, w) in NSET:
                    continue
                s = 0
                for r in range(rank):
                    s += U[u][r] * V[v][r] * W[w][r]
                if s != 0:
                    return False
    return True


def _verify_float(U, V, W, tol=1e-6):
    rank = len(U[0])
    for (u, v, w) in ENTRIES:
        s = 0.0
        for r in range(rank):
            s += U[u][r] * V[v][r] * W[w][r]
        if abs(s - 1.0) > tol:
            return False
    for u in range(NM):
        for v in range(MP):
            for w in range(PN):
                if (u, v, w) in NSET:
                    continue
                s = 0.0
                for r in range(rank):
                    s += U[u][r] * V[v][r] * W[w][r]
                if abs(s) > tol:
                    return False
    return True


def _make_rank39():
    """Strassen rank-7 on five 2x2 blocks + schoolbook on the last column."""
    rank = 5 * 7 + 4
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        v_e, v_f, v_g, v_h = k0, k1, 11 + k0, 11 + k1
        w00, w10, w01, w11 = 2 * k0, 2 * k0 + 1, 2 * k1, 2 * k1 + 1
        U[0][r] = 1; U[3][r] = 1; V[v_e][r] = 1; V[v_h][r] = 1; W[w00][r] = 1; W[w11][r] = 1; r += 1
        U[2][r] = 1; U[3][r] = 1; V[v_e][r] = 1; W[w10][r] = 1; W[w11][r] = -1; r += 1
        U[0][r] = 1; V[v_f][r] = 1; V[v_h][r] = -1; W[w01][r] = 1; W[w11][r] = 1; r += 1
        U[3][r] = 1; V[v_g][r] = 1; V[v_e][r] = -1; W[w00][r] = 1; W[w10][r] = 1; r += 1
        U[0][r] = 1; U[1][r] = 1; V[v_h][r] = 1; W[w00][r] = -1; W[w01][r] = 1; r += 1
        U[2][r] = 1; U[0][r] = -1; V[v_e][r] = 1; V[v_f][r] = 1; W[w11][r] = 1; r += 1
        U[1][r] = 1; U[3][r] = -1; V[v_g][r] = 1; V[v_h][r] = 1; W[w00][r] = 1; r += 1
    k2 = 10
    v_e2, v_g2 = k2, 11 + k2
    w02, w12 = 2 * k2, 2 * k2 + 1
    U[0][r] = 1; V[v_e2][r] = 1; W[w02][r] = 1; r += 1
    U[1][r] = 1; V[v_g2][r] = 1; W[w02][r] = 1; r += 1
    U[2][r] = 1; V[v_e2][r] = 1; W[w12][r] = 1; r += 1
    U[3][r] = 1; V[v_g2][r] = 1; W[w12][r] = 1; r += 1
    return U, V, W


def _try_random_exact(rng, R, tries):
    """Random integer decompositions, exact-verified. Return (U,V,W) or None."""
    for _ in range(tries):
        if _time_left() < 8:
            break
        U = rng.randint(-2, 2, size=(NM, R))
        V = rng.randint(-2, 2, size=(MP, R))
        W = rng.randint(-2, 2, size=(PN, R))
        if _verify_float(U, V, W):
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
    return None


def _try_rank_reduction(rng, U0, V0, W0, R, tries):
    """Exact rational rank reduction of the known decomposition.

    New search family: try to express R target terms as linear combinations of
    the R0 existing terms via an exact integer relation, then verify.
    """
    R0 = len(U0[0])
    for _ in range(tries):
        if _time_left() < 8:
            break
        # random integer combination matrix C (R x R0)
        C = rng.randint(-2, 2, size=(R, R0))
        # build candidate term vectors
        candU = np.zeros((NM, R))
        candV = np.zeros((MP, R))
        candW = np.zeros((PN, R))
        for r in range(R):
            for s in range(R0):
                c = C[r, s]
                if c:
                    candU[:, r] += c * np.array(U0[:, s])
                    candV[:, r] += c * np.array(V0[:, s])
                    candW[:, r] += c * np.array(W0[:, s])
        if _verify_float(candU, candV, candW):
            if _verify(candU.tolist(), candV.tolist(), candW.tolist()):
                return candU.tolist(), candV.tolist(), candW.tolist()
    return None


def construct():
    global DEADLINE
    DEADLINE = time.time() + 118

    best = None
    U39, V39, W39 = _make_rank39()
    if _verify(U39, V39, W39):
        best = (39, U39, V39, W39)
    else:  # should never happen, but keep a safe fallback
        U = [[0] * 44 for _ in range(NM)]
        V = [[0] * 44 for _ in range(MP)]
        W = [[0] * 44 for _ in range(PN)]
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][col] = 1
                    V[j * P + k][col] = 1
                    W[k * N + i][col] = 1
                    col += 1
        best = (44, U, V, W)

    rng = random.Random(20240517)

    # (a) exact rank reduction on the known 39-term decomposition
    for R in range(38, best[0] - 1, -1):
        if _time_left() < 12:
            break
        res = _try_rank_reduction(rng, U39, V39, W39, R, 4000)
        if res is not None:
            best = (R, res[0], res[1], res[2])
            break

    # (b) random exact decomposition search (cheap, integer, exact-verified)
    if _time_left() > 15:
        for R in range(best[0] - 1, 32, -1):
            if _time_left() < 12:
                break
            res = _try_random_exact(rng, R, 3000)
            if res is not None:
                best = (R, res[0], res[1], res[2])
                break

    return {"U": best[1], "V": best[2], "W": best[3]}
