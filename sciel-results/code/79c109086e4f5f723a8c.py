import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i1 in range(n):
        for i2 in range(m):
            for j1 in range(m):
                for j2 in range(p):
                    if i2 == j1:
                        i = i1 * m + i2
                        j = j1 * p + j2
                        k = i1 * p + j2
                        T[i, j, k] = 1.0
                        
    start_time = time.time()
    deadline = 110.0 # Leave 10s margin
    
    def verify(U, V, W):
        r = U.shape[1]
        T_approx = np.zeros_like(T)
        for l in range(r):
            T_approx += np.einsum('i,j,k->ijk', U[:, l], V[:, l], W[:, l])
        return np.allclose(T_approx, T, atol=1e-9)

    def search_rank(r, max_iters=50, restarts=3):
        for restart in range(restarts):
            if time.time() > deadline:
                return None
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Solve for U
                A_U = np.zeros((mp * pn, r))
                for l in range(r):
                    A_U[:, l] = (V[:, l] * W[:, l]).flatten()
                for i in range(nm):
                    b = T[i].flatten()
                    U[i, :], _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                    
                # Solve for V
                A_V = np.zeros((nm * pn, r))
                for l in range(r):
                    A_V[:, l] = (U[:, l] * W[:, l]).flatten()
                for j in range(mp):
                    b = T[:, j, :].flatten()
                    V[j, :], _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                    
                # Solve for W
                A_W = np.zeros((nm * mp, r))
                for l in range(r):
                    A_W[:, l] = (U[:, l] * V[:, l]).flatten()
                for k in range(pn):
                    b = T[:, :, k].flatten()
                    W[k, :], _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
            
            if verify(U, V, W):
                return U, V, W
        return None

    # Try ranks from 31 to 36
    for r in range(31, 37):
        if time.time() > deadline:
            break
        result = search_rank(r)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
    rank = nm * mp * pn
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i1 in range(n):
        for i2 in range(m):
            for j1 in range(m):
                for j2 in range(p):
                    if i2 == j1:
                        i = i1 * m + i2
                        j = j1 * p + j2
                        k = i1 * p + j2
                        U[i][column] = 1
                        V[j][column] = 1
                        W[k][column] = 1
                        column += 1
    return {"U": U, "V": V, "W": W}
