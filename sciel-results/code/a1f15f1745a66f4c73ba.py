import time
import random
import math
import numpy as np

DIMENSIONS = [3, 4, 4]

_DEADLINE = time.time() + 150.0
_SAFE_MARGIN = 5.0

def _target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _objective(U, V, W, t):
    # smooth epigraph: sum_c (U_c . V_c . W_c) approximates T, penalize t
    n, m, p = DIMENSIONS
    T = _target_tensor()
    # reshape factors: U (n*m, r), V (m*p, r), W (p*n, r)
    r = U.shape[1]
    U2 = U.reshape(n, m, r)
    V2 = V.reshape(m, p, r)
    W2 = W.reshape(p, n, r)
    # C[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    C = np.einsum('ijc,jkc,kic->ijk', U2, V2, W2)
    resid = (C - T)
    # smooth max of |resid| via log-sum-exp
    M = 50.0
    a = M * np.abs(resid)
    a_max = a.max()
    logsum = a_max + np.log(np.exp(a - a_max).sum())
    smooth_max = (logsum) / M
    return smooth_max + t

def _grad(U, V, W, t):
    n, m, p = DIMENSIONS
    T = _target_tensor()
    r = U.shape[1]
    U2 = U.reshape(n, m, r)
    V2 = V.reshape(m, p, r)
    W2 = W.reshape(p, n, r)
    C = np.einsum('ijc,jkc,kic->ijk', U2, V2, W2)
    resid = C - T
    M = 50.0
    a = M * np.abs(resid)
    a_max = a.max()
    # softmax weights
    s = np.exp(a - a_max)
    Z = s.sum()
    # d smooth_max / d resid[i,j,k] = M * sign(resid) * s[i,j,k]/Z
    dsig = M * np.sign(resid) * (s / Z)
    # grad w.r.t. U2[i,j,c] = dsig[i,j,k] * V2[j,k,c] * W2[k,i,c]
    gU2 = np.einsum('ijk,jkc,kic->ijc', dsig, V2, W2)
    gV2 = np.einsum('ijk,ijc,kic->jkc', dsig, U2, W2)
    gW2 = np.einsum('ijk,ijc,jkc->kic', dsig, U2, V2)
    return gU2.reshape(n*m, r), gV2.reshape(m*p, r), gW2.reshape(p*n, r), 1.0

def _local_search(U, V, W, t, max_iters, step, tol, deadline):
    n, m, p = DIMENSIONS
    r = U.shape[1]
    best = _objective(U, V, W, t)
    U = U.copy(); V = V.copy(); W = W.copy()
    for it in range(max_iters):
        if time.time() > deadline:
            break
        gU, gV, gW, gt = _grad(U, V, W, t)
        # adaptive step
        step_size = step
        # backtracking
        for _ in range(20):
            nU = U - step_size * gU
            nV = V - step_size * gV
            nW = W - step_size * gW
            nt = t - step_size * gt
            nv = _objective(nU, nV, nW, nt)
            if nv < best:
                U, V, W, t = nU, nV, nW, nt
                best = nv
                break
            step_size *= 0.5
        if step_size < 1e-12:
            break
        if best < tol:
            break
    return U, V, W, t, best

def _round_to_rational(U, V, W, t, tol):
    # try to snap near-half-integer entries to exact rationals
    def snap(M):
        out = M.copy()
        for idx in np.argwhere(np.abs(M) > 1e-9):
            a, b = idx
            val = M[a, b]
            # nearest half-integer
            h = np.round(2*val)/2
            if abs(val - h) < tol:
                out[a, b] = h
        return out
    return snap(U), snap(V), snap(W)

def _verify(U, V, W, tol=1e-7):
    n, m, p = DIMENSIONS
    T = _target_tensor()
    r = U.shape[1]
    U2 = U.reshape(n, m, r)
    V2 = V.reshape(m, p, r)
    W2 = W.reshape(p, n, r)
    C = np.einsum('ijc,jkc,kic->ijk', U2, V2, W2)
    return np.max(np.abs(C - T)) < tol

def _try_rank(r, time_budget, seed):
    rng = random.Random(seed)
    n, m, p = DIMENSIONS
    deadline = time.time() + time_budget
    best_obj = float('inf')
    best = None
    # number of restarts based on budget
    n_restarts = max(1, int(time_budget // 3.0))
    for rs in range(n_restarts):
        if time.time() > deadline:
            break
        # random init: small random + some structured
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        t = 1.0
        # local search
        U2, V2, W2, t2, obj = _local_search(U, V, W, t, max_iters=200, step=0.05, tol=1e-6, deadline=deadline)
        if obj < best_obj:
            best_obj = obj
            best = (U2, V2, W2, t2)
        if best_obj < 1e-6:
            break
    if best is None:
        return None
    U, V, W, t = best
    if not _verify(U, V, W, tol=1e-5):
        return None
    # polish with smaller tol
    U, V, W, t, obj = _local_search(U, V, W, t, max_iters=100, step=0.01, tol=1e-9, deadline=deadline)
    if not _verify(U, V, W, tol=1e-7):
        return None
    # snap to half-integers
    U, V, W = _round_to_rational(U, V, W, tol=1e-4)
    if not _verify(U, V, W, tol=1e-6):
        return None
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # fallback schoolbook
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    deadline = _DEADLINE - _SAFE_MARGIN
    # theoretical lower bound via flattening: max(n*p, m*p, n*m) = max(12,16,12)=16
    # schoolbook = 48. Try ranks from low to high.
    lower = max(n * p, m * p, n * m)
    # cap search to reasonable range; we must finish in time
    max_rank_to_try = min(48, lower + 12)
    best_result = None
    for r in range(lower, max_rank_to_try + 1):
        if time.time() > deadline:
            break
        # time budget per rank: more for lower ranks (more likely to fail)
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        # allocate: lower ranks get more restarts
        budget = min(remaining - 1.0, 20.0)
        if budget < 2.0:
            break
        res = _try_rank(r, budget, seed=r * 1000 + 7)
        if res is not None:
            U, V, W = res
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    if best_result is None:
        return schoolbook()
    return best_result
