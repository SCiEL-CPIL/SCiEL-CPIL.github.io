import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U0, V0, W0 = schoolbook()
    
    def check_decomp(U, V, W, tol=1e-9):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol
    
    def try_rank(r, max_iters=30, n_trials=5, deadline=None):
        best_res = np.inf
        best_U, best_V, best_W = None, None, None
        
        for trial in range(n_trials):
            if deadline and time.time() > deadline:
                break
            U = np.random.randn(nm, r) * 0.3
            V = np.random.randn(mp, r) * 0.3
            W = np.random.randn(pn, r) * 0.3
            
            for it in range(max_iters):
                if deadline and time.time() > deadline:
                    break
                
                M = (V.T * W.T).T; Mf = M.reshape(r, mp*pn); Tf = T.reshape(nm, mp*pn)
                try: U = np.linalg.solve(Mf@Mf.T, Tf@Mf.T).T
                except: U = np.linalg.lstsq(Mf.T, Tf.T, rcond=None)[0].T
                
                M = (U.T * W.T).T; Mf = M.reshape(r, nm*pn); Tf2 = T.reshape(mp, nm*pn)
                try: V = np.linalg.solve(Mf@Mf.T, Tf2@Mf.T).T
                except: V = np.linalg.lstsq(Mf.T, Tf2.T, rcond=None)[0].T
                
                M = (U.T * V.T).T; Mf = M.reshape(r, nm*mp); Tf3 = T.reshape(pn, nm*mp)
                try: W = np.linalg.solve(Mf@Mf.T, Tf3@Mf.T).T
                except: W = np.linalg.lstsq(Mf.T, Tf3.T, rcond=None)[0].T
                
                res = np.max(np.abs(np.einsum('ar,br,cr->abc', U, V, W) - T))
                if res < best_res:
                    best_res = res
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if res < 1e-10: break
            
            if best_res < 1e-10: break
        
        if best_res < 1e-10:
            return best_U, best_V, best_W, best_res
        return None, None, None, best_res
    
    best = None
    t0 = time.time()
    dl = t0 + 110
    
    for r in range(36, 15, -1):
        if time.time() > dl: break
        U, V, W, res = try_rank(r, 25, 3, dl)
        if U is not None and res < 1e-9:
            Ur, Vr, Wr = np.round(U*2)/2, np.round(V*2)/2, np.round(W*2)/2
            if check_decomp(Ur, Vr, Wr):
                best = (Ur, Vr, Wr, r); break
            else:
                best = (U, V, W, r); break
    
    if best is None:
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W, r = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
