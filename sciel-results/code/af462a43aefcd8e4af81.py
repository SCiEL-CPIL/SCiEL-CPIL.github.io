"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook
    rank_sb = n*m*p
    U = np.zeros((n*m, rank_sb))
    V = np.zeros((m*p, rank_sb))
    W = np.zeros((p*n, rank_sb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1; c += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_sb
    
    def valid(U, V, W, tol=1e-9):
        R = np.tensordot(np.tensordot(U, V, axes=(1,1)), W, axes=(2,1))
        return np.allclose(R, T, atol=tol, rtol=0)
    
    def als(T, r, iters=200, seed=0):
        np.random.seed(seed)
        n1,n2,n3 = T.shape
        U = np.random.randn(n1,r)*0.3
        V = np.random.randn(n2,r)*0.3
        W = np.random.randn(n3,r)*0.3
        for _ in range(iters):
            if time.time() > deadline-5: break
            T1 = T.reshape(n1, n2*n3)
            VW = (V*W).reshape(r, n2*n3)
            try: U = np.linalg.solve((VW@VW.T).T, (T1@VW.T).T).T
            except: U = np.linalg.lstsq((VW@VW.T).T, (T1@VW.T).T, rcond=None)[0].T
            T2 = T.transpose(1,0,2).reshape(n2, n1*n3)
            UW = (U*W).reshape(r, n1*n3)
            try: V = np.linalg.solve((UW@UW.T).T, (T2@UW.T).T).T
            except: V = np.linalg.lstsq((UW@UW.T).T, (T2@UW.T).T, rcond=None)[0].T
            T3 = T.transpose(2,0,1).reshape(n3, n1*n2)
            UV = (U*V).reshape(r, n1*n2)
            try: W = np.linalg.solve((UV@UV.T).T, (T3@UV.T).T).T
            except: W = np.linalg.lstsq((UV@UV.T).T, (T3@UV.T).T, rcond=None)[0].T
        R = np.tensordot(np.tensordot(U, V, axes=(1,1)), W, axes=(2,1))
        return U, V, W, np.linalg.norm(R-T)
    
    for r in range(20, best_rank):
        if time.time() > deadline-20: break
        for s in range(8):
            if time.time() > deadline-20: break
            U,V,W,err = als(T, r, 200, seed=s)
            if err < 1e-8 and valid(U,V,W):
                best_rank = r; best_U, best_V, best_W = U,V,W
                break
        if best_rank == r: break
    
    # Round to half-integers
    Ur = np.round(best_U*2)/2; Vr = np.round(best_V*2)/2; Wr = np.round(best_W*2)/2
    if valid(Ur, Vr, Wr, 1e-10):
        best_U, best_V, best_W = Ur, Vr, Wr
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

