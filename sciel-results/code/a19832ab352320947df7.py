import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 145
    
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check(U, V, W, tol=1e-9):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.abs(recon - T).max() < tol
    
    def als(rank, seed=0, max_iter=3000):
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, rank) * 0.5
        V = rng.randn(mp, rank) * 0.5
        W = rng.randn(pn, rank) * 0.5
        
        T1 = T.reshape(nm, -1)
        T2 = np.zeros((mp, nm*pn))
        T3 = np.zeros((pn, nm*mp))
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    T2[b, a*pn+c] = T[a,b,c]
                    T3[c, a*mp+b] = T[a,b,c]
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            VW = np.einsum('br,cr->bcr', V, W).reshape(-1, rank)
            U = T1 @ np.linalg.pinv(VW.T)
            WU = np.einsum('cr,ar->car', W, U).reshape(-1, rank)
            V = T2 @ np.linalg.pinv(WU.T)
            UV = np.einsum('ar,br->abr', U, V).reshape(-1, rank)
            W = T3 @ np.linalg.pinv(UV.T)
            if it % 100 == 0:
                err = np.abs(np.einsum('ar,br,cr->abc', U, V, W) - T).max()
                if err < 1e-12:
                    break
        
        err = np.abs(np.einsum('ar,br,cr->abc', U, V, W) - T).max()
        if err < 1e-9:
            return U, V, W
        return None
    
    for rank in range(30, 61):
        if time.time() > deadline:
            break
        for seed in range(3):
            if time.time() > deadline:
                break
            res = als(rank, seed)
            if res:
                U, V, W = res
                Ur, Vr, Wr = np.round(U*2)/2, np.round(V*2)/2, np.round(W*2)/2
                if check(Ur, Vr, Wr):
                    return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
                if check(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    rank = 60
    U = [[0]*rank for _ in range(nm)]
    V = [[0]*rank for _ in range(mp)]
    W = [[0]*rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
