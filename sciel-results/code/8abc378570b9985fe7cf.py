import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _build_tensor(n, m, p):
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _schoolbook(n, m, p):
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for r in range(rank):
        k = r % p
        j = (r // p) % m
        i = r // (p*m)
        U[i*m+j, r] = 1.0
        V[j*p+k, r] = 1.0
        W[k*n+i, r] = 1.0
    return U, V, W

def _check(U, V, W, n, m, p, tol=1e-7):
    for i in range(n):
        for k in range(p):
            for j in range(m):
                if abs(np.dot(U[i*m+j], V[j*p+k]*W[k*n+i]) - 1.0) > tol:
                    return False
    return True

def _als_solve(n, m, p, rank, T, deadline, iters=80):
    nm, mp, pn = n*m, m*p, p*n
    np.random.seed(12345)
    U = np.random.randn(nm, rank) * 0.1
    V = np.random.randn(mp, rank) * 0.1
    W = np.random.randn(pn, rank) * 0.1
    
    Tvec = T.reshape(nm, mp*pn)
    TV = np.stack([T[:,b,:].reshape(nm*pn) for b in range(mp)])
    TW = np.stack([T[:,:,c].reshape(nm*mp) for c in range(pn)])
    
    for _ in range(iters):
        if time.time() > deadline - 5:
            break
        MU = (V.T @ W).T.reshape(rank, mp*pn) if rank > 0 else None
        # Actually: MU[r, b*pn+c] = V[b,r]*W[c,r]
        MU = np.zeros((rank, mp*pn))
        for r in range(rank):
            MU[r] = (V[:,r][:,None] * W[:,r][None,:]).flatten()
        U = np.linalg.lstsq(MU, Tvec, rcond=None)[0]
        
        MV = np.zeros((rank, nm*pn))
        for r in range(rank):
            MV[r] = (U[:,r][:,None] * W[:,r][None,:]).reshape(nm,pn).flatten()
        V = np.linalg.lstsq(MV, TV, rcond=None)[0]
        
        MW = np.zeros((rank, nm*mp))
        for r in range(rank):
            MW[r] = (U[:,r][:,None] * V[:,r][None,:]).reshape(nm,mp).flatten()
        W = np.linalg.lstsq(MW, TW, rcond=None)[0]
        
        if _check(U, V, W, n, m, p, tol=1e-5):
            break
    
    if _check(U, V, W, n, m, p):
        return U, V, W
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 130.0
    n, m, p = DIMENSIONS
    sr = n*m*p
    
    Us, Vs, Ws = _schoolbook(n, m, p)
    T = _build_tensor(n, m, p)
    
    best = (sr, Us, Vs, Ws)
    
    # Try a few ranks below schoolbook
    for r in [sr-1, sr-5, sr-10, 90, 85, 80]:
        if time.time() > deadline - 25:
            break
        res = _als_solve(n, m, p, r, T, deadline, iters=60)
        if res is not None:
            best = (r, res[0], res[1], res[2])
            break
    
    br, bU, bV, bW = best
    
    # Round if close to integers
    bU = np.where(np.abs(bU - np.round(bU)) < 1e-5, np.round(bU), bU)
    bV = np.where(np.abs(bV - np.round(bV)) < 1e-5, np.round(bV), bV)
    bW = np.where(np.abs(bW - np.round(bW)) < 1e-5, np.round(bW), bW)
    
    return {"U": bU.tolist(), "V": bV.tolist(), "W": bW.tolist()}
