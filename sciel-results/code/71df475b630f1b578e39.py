import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def _verify(U, V, W, n=4, m=4, p=5, atol=1e-9):
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        for j in range(m):
            T[:, j, :] += np.outer(uc[:, j], vc[j, :]) * wc.T
    return np.allclose(T, 1.0, atol=atol)

def _round_half(M):
    return np.round(2 * np.array(M, dtype=np.float64)) / 2.0

def _strassen_terms():
    return [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _kron_strassen(n, m, p):
    U_cols, V_cols, W_cols = [], [], []
    st = _strassen_terms()
    for (u1s, v1s, w1s) in st:
        for (u2s, v2s, w2s) in st:
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            for (i1, j1), val1 in [(idx, val) for idx, val in u1s]:
                for (i2, j2), val2 in [(idx, val) for idx, val in u2s]:
                    u[(i1*2+i2)*m + (j1*2+j2)] = val1*val2
            for (j1, k1), val1 in [(idx, val) for idx, val in v1s]:
                for (j2, k2), val2 in [(idx, val) for idx, val in v2s]:
                    v[(j1*2+j2)*p + (k1*2+k2)] = val1*val2
            for (k1, i1), val1 in [(idx, val) for idx, val in w1s]:
                for (k2, i2), val2 in [(idx, val) for idx, val in w2s]:
                    w[(k1*2+k2)*n + (i1*2+i2)] = val1*val2
            U_cols.append(u); V_cols.append(v); W_cols.append(w)
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[i*m+j] = 1
            v = np.zeros(m*p); v[j*p+4] = 1
            w = np.zeros(p*n); w[4*n+i] = 1
            U_cols.append(u); V_cols.append(v); W_cols.append(w)
    return np.array(U_cols).T.tolist(), np.array(V_cols).T.tolist(), np.array(W_cols).T.tolist()

def _als(U, V, W, n, m, p, rank, max_iters=300, deadline=None, tol=1e-13):
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        # Optimize U
        for i in range(n):
            for j in range(m):
                A = np.array([V[j*p+k] * W[k*n+i] for k in range(p)])
                b = np.ones(p)
                try: U[i*m+j] = np.linalg.lstsq(A, b, rcond=None)[0]
                except: pass
        # Optimize V
        for j in range(m):
            for k in range(p):
                A = np.array([U[i*m+j] * W[k*n+i] for i in range(n)])
                b = np.ones(n)
                try: V[j*p+k] = np.linalg.lstsq(A, b, rcond=None)[0]
                except: pass
        # Optimize W
        for i in range(n):
            for k in range(p):
                A = np.array([U[i*m+j] * V[j*p+k] for j in range(m)])
                b = np.ones(m)
                try: W[k*n+i] = np.linalg.lstsq(A, b, rcond=None)[0]
                except: pass
        # Error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] for r in range(rank))
                    err += (s-1.0)**2
        err = np.sqrt(err)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < tol:
            break
    return best_U, best_V, best_W, best_err

def _try_rank(U_base, V_base, W_base, n, m, p, target_rank, deadline, seeds=3):
    Ua = np.array(U_base, dtype=np.float64)
    Va = np.array(V_base, dtype=np.float64)
    Wa = np.array(W_base, dtype=np.float64)
    base_rank = Ua.shape[1]
    
    for seed in range(seeds):
        if time.time() > deadline:
            return None
        np.random.seed(seed*7919 + target_rank)
        if seed == 0:
            idx = np.random.choice(base_rank, target_rank, replace=False)
            Ui, Vi, Wi = Ua[:,idx], Va[:,idx], Wa[:,idx]
        elif seed == 1:
            idx = np.random.choice(base_rank, target_rank, replace=False)
            Ui = Ua[:,idx] + np.random.randn(n*m, target_rank)*0.05
            Vi = Va[:,idx] + np.random.randn(m*p, target_rank)*0.05
            Wi = Wa[:,idx] + np.random.randn(p*n, target_rank)*0.05
        else:
            Ui = np.random.randn(n*m, target_rank)*0.3
            Vi = np.random.randn(m*p, target_rank)*0.3
            Wi = np.random.randn(p*n, target_rank)*0.3
        
        Ub, Vb, Wb, err = _als(Ui, Vi, Wi, n, m, p, target_rank, 
                                max_iters=200, deadline=deadline, tol=1e-12)
        Ur = _round_half(Ub); Vr = _round_half(Vb); Wr = _round_half(Wb)
        if _verify(Ur.tolist(), Vr.tolist(), Wr.tolist(), n, m, p):
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    deadline = start + 140
    n, m, p = DIMENSIONS
    
    # Baseline: Kronecker Strassen (rank 65)
    U_k, V_k, W_k = _kron_strassen(n, m, p)
    if not _verify(U_k, V_k, W_k, n, m, p):
        return _schoolbook()
    
    best = {"U": U_k, "V": V_k, "W": W_k}
    best_rank = len(U_k[0])
    
    # Try reducing rank
    for tr in range(best_rank - 1, max(45, best_rank - 20), -1):
        if time.time() > deadline - 30:
            break
        res = _try_rank(U_k, V_k, W_k, n, m, p, tr, deadline - 30, seeds=4)
        if res:
            best = {"U": res[0], "V": res[1], "W": res[2]}
            best_rank = tr
    
    # Try random inits at lower ranks
    for tr in range(best_rank - 1, max(45, best_rank - 15), -1):
        if time.time() > deadline - 15:
            break
        found = False
        for seed in range(10):
            if time.time() > deadline - 15:
                break
            np.random.seed(seed*104729 + tr)
            Ui = np.random.randn(n*m, tr)*0.4
            Vi = np.random.randn(m*p, tr)*0.4
            Wi = np.random.randn(p*n, tr)*0.4
            Ub, Vb, Wb, err = _als(Ui, Vi, Wi, n, m, p, tr,
                                    max_iters=250, deadline=deadline-15, tol=1e-12)
            Ur = _round_half(Ub); Vr = _round_half(Vb); Wr = _round_half(Wb)
            if _verify(Ur.tolist(), Vr.tolist(), Wr.tolist(), n, m, p):
                best = {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
                best_rank = tr
                found = True
                break
        if found:
            continue
    
    return best
