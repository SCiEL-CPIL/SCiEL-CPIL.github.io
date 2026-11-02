import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 7]

def _build_rank39():
    """Return (U,V,W) as list-of-lists for the known rank-39 hybrid decomposition."""
    n, m, p = 2, 3, 7
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    terms = []
    def add_term(u, v, w):
        ul = [0.0]*(n*m); vl = [0.0]*(m*p); wl = [0.0]*(p*n)
        for idx, val in u.items(): ul[idx] = val
        for idx, val in v.items(): vl[idx] = val
        for idx, val in w.items(): wl[idx] = val
        terms.append((ul, vl, wl))

    def strassen_block(b0):
        # 2x2 (A cols 0,1) times 2x2 (B cols b0, b0+1)
        M1 = ({u_idx(0,0):1,u_idx(1,1):1},{v_idx(0,b0):1,v_idx(1,b0+1):1},{w_idx(b0,0):1,w_idx(b0+1,1):1})
        M2 = ({u_idx(1,0):1,u_idx(1,1):1},{v_idx(0,b0):1},{w_idx(b0,1):1,w_idx(b0+1,1):-1})
        M3 = ({u_idx(0,0):1},{v_idx(0,b0+1):1,v_idx(1,b0+1):-1},{w_idx(b0+1,0):1,w_idx(b0+1,1):1})
        M4 = ({u_idx(1,1):1},{v_idx(1,b0):1,v_idx(0,b0):-1},{w_idx(b0,0):1,w_idx(b0,1):1})
        M5 = ({u_idx(0,0):1,u_idx(0,1):1},{v_idx(1,b0+1):1},{w_idx(b0,0):-1,w_idx(b0+1,0):1})
        M6 = ({u_idx(1,0):1,u_idx(0,0):-1},{v_idx(0,b0):1,v_idx(0,b0+1):1},{w_idx(b0+1,1):1})
        M7 = ({u_idx(0,1):1,u_idx(1,1):-1},{v_idx(1,b0):1,v_idx(1,b0+1):1},{w_idx(b0,0):1})
        for (u,v,w) in (M1,M2,M3,M4,M5,M6,M7):
            add_term(u,v,w)

    # Strassen for col blocks (0,1),(2,3),(4,5)
    strassen_block(0)
    strassen_block(2)
    strassen_block(4)
    # Schoolbook for col 6, k in {0,1}
    for k in (0,1):
        for i in (0,1):
            add_term({u_idx(i,k):1},{v_idx(k,6):1},{w_idx(6,i):1})
    # Schoolbook for k=2, all cols
    for k in range(p):
        for i in (0,1):
            add_term({u_idx(i,2):1},{v_idx(2,k):1},{w_idx(k,i):1})

    rank = len(terms)
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    for c,(ul,vl,wl) in enumerate(terms):
        for i in range(n*m): U[i][c]=ul[i]
        for i in range(m*p): V[i][c]=vl[i]
        for i in range(p*n): W[i][c]=wl[i]
    return U, V, W


def _tensor_from(U,V,W):
    n, m, p = 2, 3, 7
    r = len(U[0])
    T = np.zeros((n, m, p))
    Ua = np.array(U); Va = np.array(V); Wa = np.array(W)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                T[i,j,k] = s
    return T


def _target_tensor():
    n, m, p = 2, 3, 7
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j == k:
                    T[i,j,k] = 1.0
    return T


def _verify(U, V, W, tol=1e-9):
    T = _tensor_from(U, V, W)
    Tt = _target_tensor()
    return np.max(np.abs(T - Tt)) < tol


def _snap_halfint(x):
    return np.round(2.0 * x) / 2.0


def _residual(U, V, W):
    T = _tensor_from(U, V, W)
    Tt = _target_tensor()
    return float(np.sum((T - Tt)**2))


def _search_rank(r, U0, V0, W0, time_budget, rng):
    """Try to find a rank-r decomposition. U0/V0/W0 are the rank-39 arrays (used for init of first r cols)."""
    n, m, p = 2, 3, 7
    du, dv, dw = n*m, m*p, p*n
    best = None
    best_res = float('inf')
    start = time.time()

    # initialize: take first r columns of the rank-39 solution
    U = np.array(U0)[:, :r].copy()
    V = np.array(V0)[:, :r].copy()
    W = np.array(W0)[:, :r].copy()

    def cur_res():
        return _residual(U.tolist(), V.tolist(), W.tolist())

    res = cur_res()
    if res < 1e-12:
        return (U, V, W, res)

    # local search: coordinate-descent-ish with random restarts
    for restart in range(200):
        if time.time() - start > time_budget:
            break
        U = np.array(U0)[:, :r].copy()
        V = np.array(V0)[:, :r].copy()
        W = np.array(W0)[:, :r].copy()
        # random perturbation
        scale = 0.3
        U += rng.uniform(-scale, scale, U.shape)
        V += rng.uniform(-scale, scale, V.shape)
        W += rng.uniform(-scale, scale, W.shape)
        res = cur_res()
        lr = 0.05
        for it in range(3000):
            if time.time() - start > time_budget:
                break
            # gradient of residual w.r.t. U[i,c]: d/dU[i,c] sum (T-Tt)^2
            # T[i,j,k] = sum_c' U[i,j,c'] V[j,k,c'] W[k,i,c']
            # dT[i,j,k]/dU[i,j,c] = V[j,k,c] W[k,i,c]
            # grad_U[i*m+j, c] = sum_{j,k} 2 (T[i,j,k]-Tt[i,j,k]) V[j*p+k,c] W[k*n+i,c]
            T = _tensor_from(U.tolist(), V.tolist(), W.tolist())
            Tt = _target_tensor()
            diff = T - Tt
            gU = np.zeros_like(U); gV = np.zeros_like(V); gW = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        d = diff[i,j,k]
                        if abs(d) < 1e-15: continue
                        for c in range(r):
                            gU[i*m+j, c] += 2*d*V[j*p+k,c]*W[k*n+i,c]
                            gV[j*p+k, c] += 2*d*U[i*m+j,c]*W[k*n+i,c]
                            gW[k*n+i, c] += 2*d*U[i*m+j,c]*V[j*p+k,c]
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
            res = cur_res()
            if res < best_res:
                best_res = res
                best = (U.copy(), V.copy(), W.copy())
            if res < 1e-12:
                break
            # adapt lr
            if it % 100 == 0:
                lr *= 0.9

    # snap to half-integers and verify
    if best is not None:
        U2, V2, W2 = best
        Us = _snap_halfint(U2).tolist()
        Vs = _snap_halfint(V2).tolist()
        Ws = _snap_halfint(W2).tolist()
        if _verify(Us, Vs, Ws):
            return (Us, Vs, Ws, 0.0)
        # also try snapping the current
        Us = _snap_halfint(U).tolist(); Vs = _snap_halfint(V).tolist(); Ws = _snap_halfint(W).tolist()
        if _verify(Us, Vs, Ws):
            return (Us, Vs, Ws, 0.0)
    return None


def construct():
    start = time.time()
    deadline = start + 110  # safety margin under 120s
    rng = random.Random(12345)

    U0, V0, W0 = _build_rank39()
    # sanity check fallback
    assert _verify(U0, V0, W0), "rank-39 construction failed"

    best = (U0, V0, W0)
    best_rank = 39

    # Try to improve from rank 38 downward
    for r in range(38, 20, -1):
        if time.time() > deadline - 15:
            break
        budget = min(20.0, deadline - time.time() - 5)
        if budget < 3.0:
            break
        res = _search_rank(r, U0, V0, W0, budget, rng)
        if res is not None:
            U, V, W, _ = res
            best = (U, V, W)
            best_rank = r
            # continue trying to go lower
            continue
        else:
            # could not achieve rank r; stop (ranks are monotone-ish in difficulty)
            break

    U, V, W = best
    return {"U": U, "V": V, "W": W}
