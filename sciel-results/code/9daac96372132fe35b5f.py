"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    for rank in range(30, 61):
        if time.time() > deadline - 15:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, rank, deadline):
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    max_attempts = 3
    max_iters = 80
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        np.random.seed(attempt * 7 + 13)
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            A_U = (V[None,:,:] * W[:,None,:]).reshape(-1, rank)
            A_V = (U[None,:,:] * W[:,None,:]).reshape(-1, rank)
            A_W = (U[:,None,:] * V[None,:,:]).reshape(-1, rank)
            
            for a in range(nm):
                U[a] = np.linalg.lstsq(A_U, T[a].reshape(-1), rcond=None)[0]
            for b in range(mp):
                V[b] = np.linalg.lstsq(A_V, T[:,b,:].reshape(-1), rcond=None)[0]
            for c in range(pn):
                W[c] = np.linalg.lstsq(A_W, T[:,:,c].reshape(-1), rcond=None)[0]
            
            if it % 10 == 9:
                if _check(U, V, W, T):
                    Ur, Vr, Wr = np.round(U*2)/2, np.round(V*2)/2, np.round(W*2)/2
                    if _check(Ur, Vr, Wr, T):
                        return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
    return None

def _check(U, V, W, T):
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if abs(np.dot(U[a], V[b]*W[c]) - T[a,b,c]) > 1e-6:
                    return False
    return True

# EVOLVE-BLOCK-END
