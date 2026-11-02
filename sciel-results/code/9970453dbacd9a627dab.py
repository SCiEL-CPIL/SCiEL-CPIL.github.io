"""Rewritten program: Strassen-per-block (rank 35) with a time-boxed PSLQ global-dependency search."""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

_DEADLINE = 110.0  # safety margin before 150s kill

def _true_tensor(n, m, p):
    # T[i, k, j] = 1 for all (the matmul coefficient tensor)
    T = np.ones((n, p, m), dtype=object)
    return T

def _reconstruct(U, V, W, n, m, p):
    # T[i,k,j] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    r = U.shape[1]
    T = np.zeros((n, p, m), dtype=object)
    U = np.asarray(U); V = np.asarray(V); W = np.asarray(W)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                T[i,k,j] = s
    return T

def _strassen_block(n, m, p):
    # Build rank-35 Strassen decomposition.
    n, m, p = 2, 2, 10
    # monomial space index: a_idx*4 + b_idx, a_idx,b_idx in 0..3 for (00,01,10,11)
    def aidx(i, j): return i*2 + j
    def bidx(j, c): return j*2 + c
    # M terms: list of (A_dict {(i,j):coef}, B_dict {(j,c):coef})
    Ms = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1,(0,1):1}),
        ({(0,0):1,(0,1):1}, {(1,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}),
        ({(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
    ]
    # Build monomial vectors for the 7 M's (length 8)
    Mvec = []
    for A, B in Ms:
        v = [0]*8
        for (i,j),a in A.items():
            for (jj,c),b in B.items():
                v[aidx(i,j)*4 + bidx(jj,c)] += a*b
        Mvec.append(v)
    Mmat = np.array(Mvec, dtype=float).T  # 8 x 7
    # Targets for C_{i, c}
    targets = {}
    targets[(0,0)] = [0]*8; targets[(0,0)][aidx(0,0)*4+bidx(0,0)] = 1; targets[(0,0)][aidx(0,1)*4+bidx(1,0)] = 1
    targets[(0,1)] = [0]*8; targets[(0,1)][aidx(0,0)*4+bidx(0,1)] = 1; targets[(0,1)][aidx(0,1)*4+bidx(1,1)] = 1
    targets[(1,0)] = [0]*8; targets[(1,0)][aidx(1,0)*4+bidx(0,0)] = 1; targets[(1,0)][aidx(1,1)*4+bidx(1,0)] = 1
    targets[(1,1)] = [0]*8; targets[(1,1)][aidx(1,0)*4+bidx(0,1)] = 1; targets[(1,1)][aidx(1,1)*4+bidx(1,1)] = 1
    dcoef = {}
    for key, tv in targets.items():
        d, res, rank, sv = np.linalg.lstsq(Mmat, np.array(tv, dtype=float), rcond=None)
        d = np.round(d).astype(int)
        dcoef[key] = d
    # Build U, V, W
    rank = 7 * (p // 2)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for b in range(p // 2):
        for (A, B) in Ms:
            for (i, j), a in A.items():
                U[i*m + j][col] += a
            for (j, c), bv in B.items():
                V[j*p + (2*b + c)][col] += bv
            for (i, c) in [(0,0),(0,1),(1,0),(1,1)]:
                d = dcoef[(i, c)]
                if d is not None:
                    W[(2*b + c)*n + i][col] += d
            col += 1
    return {"U": U, "V": V, "W": W}

def _verify(res, n, m, p):
    T = _reconstruct(res["U"], res["V"], res["W"], n, m, p)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                if T[i,k,j] != 1:
                    return False
    return True

def _pslq_search(n, m, p, base_rank, deadline):
    # Time-boxed global dependency search; returns better decomposition or None.
    import random
    rng = random.Random(12345)
    best = None
    for r in range(max(2, base_rank - 1), 1, -1):
        if time.time() > deadline:
            break
        # A few random integer attempts
        for attempt in range(3):
            if time.time() > deadline:
                break
            U = [[rng.randint(-2,2) for _ in range(r)] for _ in range(n*m)]
            V = [[rng.randint(-2,2) for _ in range(r)] for _ in range(m*p)]
            # Solve for W: for each (k,i) row, the linear system over (i,k,j)
            # Build linear system: unknowns W[k*n+i, c] for all k,i,c (p*n*r unknowns)
            # equations: for each (i,k,j): sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
            rows = []
            colidx = []
            rhs = []
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        row = [0]*(p*n*r)
                        for c in range(r):
                            coef = U[i*m+j][c] * V[j*p+k][c]
                            if coef != 0:
                                row[(k*n + i)*r + c] = coef
                        rows.append(row)
                        rhs.append(1)
            A = np.array(rows, dtype=float)
            b = np.array(rhs, dtype=float)
            sol, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            sol = np.round(sol).astype(int)
            # verify
            ok = True
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        s = 0
                        for c in range(r):
                            s += U[i*m+j][c]*V[j*p+k][c]*sol[(k*n+i)*r + c]
                        if s != 1:
                            ok = False
                            break
                    if not ok: break
                if not ok: break
            if ok:
                W = [[0]*r for _ in range(p*n)]
                for k in range(p):
                    for i in range(n):
                        for c in range(r):
                            W[k*n+i][c] = sol[(k*n+i)*r + c]
                best = {"U": U, "V": V, "W": W}
                return best
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = _DEADLINE
    base = _strassen_block(n, m, p)
    if not _verify(base, n, m, p):
        # Fallback to schoolbook (guaranteed valid)
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        base = {"U": U, "V": V, "W": W}
    base_rank = base["U"].shape[1]
    # Attempt PSLQ global search for lower rank
    better = _pslq_search(n, m, p, base_rank, deadline)
    if better is not None and _verify(better, n, m, p):
        return better
    return base
