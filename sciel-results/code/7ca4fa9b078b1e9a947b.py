import numpy as np

def construct():
    n, m, p = 2, 4, 5
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a low-rank decomposition
    for rank in range(20, 35):
        for trial in range(30):
            np.random.seed(trial * 1000 + rank)
            U = np.round(np.random.randn(n*m, rank)) / 2
            V = np.round(np.random.randn(m*p, rank)) / 2
            W = np.round(np.random.randn(p*n, rank)) / 2
            
            for _ in range(20):
                for r in range(rank):
                    vw = np.outer(V[:,r], W[:,r])
                    u = np.array([np.sum(T[a]*vw) for a in range(n*m)])
                    ns = np.sum(vw*vw)
                    if ns > 1e-10: U[:,r] = np.round(u/ns*2)/2
                
                for r in range(rank):
                    uw = np.outer(U[:,r], W[:,r])
                    v = np.array([np.sum(T[:,b,:]*uw) for b in range(m*p)])
                    ns = np.sum(uw*uw)
                    if ns > 1e-10: V[:,r] = np.round(v/ns*2)/2
                
                for r in range(rank):
                    uv = np.outer(U[:,r], V[:,r])
                    w = np.array([np.sum(T[:,:,c]*uv) for c in range(p*n)])
                    ns = np.sum(uv*uv)
                    if ns > 1e-10: W[:,r] = np.round(w/ns*2)/2
                
                if np.allclose(np.einsum('ar,br,cr->abc', U, V, W), T, atol=1e-10):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
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
