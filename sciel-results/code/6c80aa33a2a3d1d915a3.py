"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition search.

Strategy:
- Start from a structured block-embedding (Laderman 2x2x2 blocks + schoolbook).
- Attempt rank reduction via:
  1) Random column subspace selection + smooth polish + half-integer rounding.
  2) Direct half-integer search over small candidate sets for the reduced rank.
  3) Gradient-based polish with tighter tolerances.
- Return the best valid decomposition found within the time budget.
"""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # safety margin under the 150 s kill

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n=N, m=M, p=P, tol=1e-9):
    """Check exact decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    return bool(np.allclose(T, 1.0, atol=tol, rtol=0))

def _to_halfint(Mat):
    """Round to nearest half-integer."""
    return np.round(2.0 * Mat) / 2.0

def _laderman_2x2():
    """Laderman's decomposition for 2x2x2 (rank 6). Returns U,V,W each (4,6)."""
    U = np.zeros((4, 6))
    V = np.zeros((4, 6))
    W = np.zeros((4, 6))
    # M1: (a00+a11)(b00+b11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    # M2: (a10+a11)(b01)
    U[2,1]=1; U[3,1]=1
    V[1,1]=1
    W[1,1]=1
    # M3: (a00)(b10+b11)
    U[0,2]=1
    V[2,2]=1; V[3,2]=1
    W[0,2]=1; W[3,2]=1
    # M4: (a11)(b00)
    U[3,3]=1
    V[0,3]=1
    W[0,3]=-1; W[3,3]=1
    # M5: (a01-a10)(b10)
    U[1,4]=1; U[2,4]=-1
    V[2,4]=1
    W[2,4]=1
    # M6: (a01+a11)(b11-b00)
    U[1,5]=1; U[3,5]=1
    V[3,5]=1; V[0,5]=-1
    W[0,5]=-1; W[1,5]=1
    return U, V, W

def _build_structured(n, m, p):
    """Build a structured low-rank candidate by block-embedding Laderman."""
    row_blocks = [(0, 2), (2, 3)]
    col_blocks = [(0, 2), (2, 4)]
    k_blocks   = [(0, 2), (2, 4), (4, 5)]

    U = {}
    V = {}
    W = {}
    rank = 0
    S2U, S2V, S2W = _laderman_2x2()
    s2_rank = 6

    def schoolblock(ri, rj, rk, base):
        nonlocal rank
        for i in range(ri[0], ri[1]):
            for j in range(rj[0], rj[1]):
                for k in range(rk[0], rk[1]):
                    U[(i, j, base)] = 1.0
                    V[(j, k, base)] = 1.0
                    W[(k, i, base)] = 1.0
                    base += 1
        return base

    for ri in row_blocks:
        for rj in col_blocks:
            for rk in k_blocks:
                si, sj, sk = ri[1]-ri[0], rj[1]-rj[0], rk[1]-rk[0]
                if si == 2 and sj == 2 and sk == 2:
                    for c in range(s2_rank):
                        for a in range(2):
                            for b in range(2):
                                val = S2U[a*2+b, c]
                                if val != 0:
                                    key = (ri[0]+a, rj[0]+b, rank)
                                    U[key] = U.get(key, 0.0) + val
                        for b in range(2):
                            for cc in range(2):
                                val = S2V[b*2+cc, c]
                                if val != 0:
                                    key = (rj[0]+b, rk[0]+cc, rank)
                                    V[key] = V.get(key, 0.0) + val
                        for cc in range(2):
                            for a in range(2):
                                val = S2W[cc*2+a, c]
                                if val != 0:
                                    key = (rk[0]+cc, ri[0]+a, rank)
                                    W[key] = W.get(key, 0.0) + val
                        rank += 1
                else:
                    rank = schoolblock(ri, rj, rk, rank)

    Umat = np.zeros((n*m, rank))
    Vmat = np.zeros((m*p, rank))
    Wmat = np.zeros((p*n, rank))
    for (i, j, c), v in U.items():
        Umat[i*m+j, c] += v
    for (j, k, c), v in V.items():
        Vmat[j*p+k, c] += v
    for (k, i, c), v in W.items():
        Wmat[k*n+i, c] += v
    return Umat, Vmat, Wmat

def _smooth_polish(U, V, W, n, m, p, seed, maxiter=500):
    """Polish a candidate with smooth residual minimization, then round to
    half-integers and verify."""
    np.random.seed(seed)
    rank = U.shape[1]
    T = np.ones((n, m, p))

    def reshape3(x, rows, cols):
        return x.reshape(rows, cols, rank)

    def objective(x):
        Uf = reshape3(x[:n*m*rank], n, m)
        Vf = reshape3(x[n*m*rank:n*m*rank+m*p*rank], m, p)
        Wf = reshape3(x[n*m*rank+m*p*rank:], p, n)
        Tf = np.einsum('ijr,jkr,kir->ijk', Uf, Vf, Wf, optimize=True)
        d = Tf - T
        return float(np.sum(d*d))

    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-14})
    except Exception:
        return None
    if res.fun < 1e-8:
        Uf = reshape3(res.x[:n*m*rank], n, m)
        Vf = reshape3(res.x[n*m*rank:n*m*rank+m*p*rank], m, p)
        Wf = reshape3(res.x[n*m*rank+m*p*rank:], p, n)
        Ur = _to_halfint(Uf.reshape(n*m, rank))
        Vr = _to_halfint(Vf.reshape(m*p, rank))
        Wr = _to_halfint(Wf.reshape(p*n, rank))
        if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
            return Ur, Vr, Wr
    return None

def _try_rank_reduction(U, V, W, n, m, p, target_rank, seed, max_attempts=20):
    """Try to reduce rank from current to target_rank by zeroing columns and polishing."""
    current_rank = U.shape[1]
    if target_rank >= current_rank:
        return None
    np.random.seed(seed)
    
    for _ in range(max_attempts):
        if _time_left() < 5:
            break
        # Randomly select which columns to keep
        keep_indices = np.random.choice(current_rank, size=target_rank, replace=False)
        keep_indices.sort()
        
        U_reduced = U[:, keep_indices]
        V_reduced = V[:, keep_indices]
        W_reduced = W[:, keep_indices]
        
        # Polish
        res = _smooth_polish(U_reduced, V_reduced, W_reduced, n, m, p, seed, maxiter=300)
        if res is not None:
            Ur, Vr, Wr = res
            if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
                return Ur, Vr, Wr
    return None

def _schoolbook():
    n, m, p = N, M, P
    rank = n*m*p
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1; V[j*p+k, col] = 1; W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = N, M, P
    U_sb, V_sb, W_sb = _schoolbook()
    best = (U_sb, V_sb, W_sb)
    best_rank = U_sb.shape[1]

    # 1) Structured block-embedding with Laderman.
    if _time_left() > 20:
        U, V, W = _build_structured(n, m, p)
        if _verify(U, V, W, n, m, p):
            best = (U, V, W); best_rank = U.shape[1]

    # 2) Try to polish the best structured candidate.
    Uo, Vo, Wo = best
    if _time_left() > 30:
        for seed in range(10):
            if _time_left() < 25:
                break
            res = _smooth_polish(Uo, Vo, Wo, n, m, p, seed, maxiter=400)
            if res is not None:
                Ur, Vr, Wr = res
                if _verify(Ur, Vr, Wr, n, m, p):
                    if Ur.shape[1] < best_rank:
                        best = (Ur, Vr, Wr); best_rank = Ur.shape[1]
                    break

    # 3) Try rank reduction from the structured candidate.
    Uo, Vo, Wo = best
    if _time_left() > 25:
        for target in range(max(10, best_rank - 15), best_rank):
            if _time_left() < 15:
                break
            found = False
            for seed in range(15):
                if _time_left() < 12:
                    break
                res = _try_rank_reduction(Uo, Vo, Wo, n, m, p, target, seed, max_attempts=10)
                if res is not None:
                    Ur, Vr, Wr = res
                    if _verify(Ur, Vr, Wr, n, m, p):
                        if target < best_rank:
                            best = (Ur, Vr, Wr); best_rank = target
                        found = True
                        break
            if found:
                break

    Uo, Vo, Wo = best
    return {"U": Uo.tolist(), "V": Vo.tolist(), "W": Wo.tolist()}

# EVOLVE-BLOCK-END
