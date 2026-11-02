"""Matrix multiplication 2x2 by 2x3 rank-minimal exact tensor decomposition search."""
import time
import math
import random
import itertools
import numpy as np

DIMENSIONS = [2, 2, 3]
DEADLINE = 142.0   # safety margin before 150s wall-clock kill
TOL = 1e-11
MAXRANK = 12
MAXINT = 100000
MAXRAT = 200000

# ---------- exact tensor (trace convention W[k*n+i, r]) ----------
def _tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

TENS = _tensor()
NM, MP, PN = TENS.shape


def _verify(U, V, W):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    # (i,j,k) coefficient = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    M1 = (U * V.T).reshape(NM, MP)
    M2 = (W.T * U).reshape(PN, NM)
    recon = np.einsum('ab,cb->ac', M1, M2)
    recon = recon.reshape(NM, MP, PN)
    return np.max(np.abs(recon - TENS)) <= TOL


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


# ---------- smooth log-sum-exp surrogate ----------
def _surrogate(U, V, W, beta):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    M1 = (U * V.T).reshape(NM, MP)
    M2 = (W.T * U).reshape(PN, NM)
    recon = np.einsum('ab,cb->ac', M1, M2).reshape(NM, MP, PN)
    d = recon - TENS
    absd = np.abs(d)
    mx = absd.max()
    if mx > 1.0 / beta:
        return math.log(math.exp(beta * mx) - 1.0) / beta
    return float(np.log1p(np.exp(beta * absd) - 1.0).sum()) / beta


# ---------- integer / rational snap ----------
def _snap(x):
    xi = int(round(x))
    if abs(x - xi) <= 1e-5:
        return xi
    for q in range(2, 13):
        num = round(x * q)
        if abs(x - num / q) <= 1e-5:
            return (num, q)
    return (int(round(x)), 1)


def _snap_to_int(x):
    xi = int(round(x))
    if abs(x - xi) <= 1e-5:
        return xi
    return None


def _snap_matrix(M):
    out = []
    for row in M:
        r = []
        for v in row:
            s = _snap(v)
            if isinstance(s, int):
                r.append(float(s))
            else:
                r.append(float(s[0]) / s[1])
        out.append(r)
    return out


# ---------- local refinement (smooth) ----------
def _refine(U, V, W, budget, beta):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    lr = r * r
    lr2 = lr * lr
    M1 = (U * V.T).reshape(NM, MP)
    M2 = (W.T * U).reshape(PN, NM)
    recon = np.einsum('ab,cb->ac', M1, M2).reshape(NM, MP, PN)
    d = recon - TENS
    ad = np.abs(d)
    mx = ad.max()
    if mx > 1.0 / beta:
        loss = math.log(math.exp(beta * mx) - 1.0) / beta
    else:
        loss = float(np.log1p(np.exp(beta * ad) - 1.0).sum()) / beta
    gradU = (M2 * V).reshape(NM, lr)
    gradV = (M1 * U).reshape(MP, lr)
    gradW = (M1 * W).reshape(PN, lr)
    G = np.concatenate([gradU.ravel(), gradV.ravel(), gradW.ravel()])
    best = (loss, U, V, W)
    step = 1e-4
    for _ in range(4000):
        if time.time() > budget:
            break
        if np.linalg.norm(G) < 1e-9:
            break
        for _ in range(30):
            cand = np.concatenate([U.ravel(), V.ravel(), W.ravel()]) - step * G
            cU = cand[:lr].reshape(NM, r)
            cV = cand[lr:lr2].reshape(MP, r)
            cW = cand[lr2:].reshape(PN, r)
            cM1 = (cU * cV.T).reshape(NM, MP)
            cM2 = (cW.T * cU).reshape(PN, NM)
            cR = np.einsum('ab,cb->ac', cM1, cM2).reshape(NM, MP, PN)
            cd = cR - TENS
            cad = np.abs(cd)
            cmx = cad.max()
            if cmx > 1.0 / beta:
                cl = math.log(math.exp(beta * cmx) - 1.0) / beta
            else:
                cl = float(np.log1p(np.exp(beta * cad) - 1.0).sum()) / beta
            if cl < loss:
                U, V, W = cU, cV, cW
                M1 = (U * V.T).reshape(NM, MP)
                M2 = (W.T * U).reshape(PN, NM)
                recon = np.einsum('ab,cb->ac', M1, M2).reshape(NM, MP, PN)
                d = recon - TENS
                ad = np.abs(d)
                mx = ad.max()
                if mx > 1.0 / beta:
                    loss = math.log(math.exp(beta * mx) - 1.0) / beta
                else:
                    loss = float(np.log1p(np.exp(beta * ad) - 1.0).sum()) / beta
                gradU = (M2 * V).reshape(NM, lr)
                gradV = (M1 * U).reshape(MP, lr)
                gradW = (M1 * W).reshape(PN, lr)
                G = np.concatenate([gradU.ravel(), gradV.ravel(), gradW.ravel()])
                if loss < best[0]:
                    best = (loss, U.copy(), V.copy(), W.copy())
                step *= 1.05
            else:
                step *= 0.5
        if time.time() > budget:
            break
    return best


def _try_rank(r, budget):
    best = None
    beta = 200.0
    nrestart = 0
    while time.time() < budget and nrestart < 400:
        nrestart += 1
        U = np.random.randn(NM, r) * 0.7
        V = np.random.randn(MP, r) * 0.7
        W = np.random.randn(PN, r) * 0.7
        sub = budget if time.time() + 0.15 < budget else time.time() + 0.05
        loss, U, V, W = _refine(U, V, W, sub, beta)
        if best is None or loss < best[0]:
            best = (loss, U, V, W)
        if loss < 1e-9:
            break
        if nrestart % 8 == 0:
            beta *= 1.4
            if beta > 4000.0:
                beta = 4000.0
    if best is None:
        return None
    _, U, V, W = best
    sU, sV, sW = _snap_matrix(U), _snap_matrix(V), _snap_matrix(W)
    if _verify(sU, sV, sW):
        return {"U": sU, "V": sV, "W": sW}
    return None


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    best = _schoolbook()
    for r in range(8, MAXRANK + 1):
        if time.time() > deadline - 3.0:
            break
        sub = min(deadline - 2.0, time.time() + 9.0)
        res = _try_rank(r, sub)
        if res is not None:
            best = res
            break
    return best
