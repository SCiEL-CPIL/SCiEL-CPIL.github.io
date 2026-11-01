import numpy as np
import time

DIMENSIONS = [2, 2, 12]

def construct():
    t0 = time.time()
    deadline = t0 + 108  # 108s deadline, 12s safety margin
    n, m, p = DIMENSIONS
    nrm, nvm, nwm = n*m, m*p, p*n  # 4, 24, 24
    
    # Build target tensor
    T = np.zeros((nrm, nvm, nwm))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find lower rank via ALS
    for r in [36, 38, 40]:
        if time.time() > deadline:
            break
        result = _als_solve(T, r, deadline)
        if result is not None:
            return result
    
    # Fallback: Strassen rank 42
    return _strassen_42(n, m, p)

def _als_solve(T, r, deadline):
    nrm, nvm, nwm = T.shape
    
    # Try warm start from Strassen (truncate)
    strassen = _strassen_42_arr(2, 2, 12)
    U_s, V_s, W_s = strassen
    
    inits = []
    # Warm start: truncate Strassen
    if r <= 42:
        inits.append((U_s[:, :r].copy(), V_s[:, :r].copy(), W_s[:, :r].copy()))
    # Random inits
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        inits.append((np.random.randn(nrm, r)*0.3, np.random.randn(nvm, r)*0.3, np.random.randn(nwm, r)*0.3))
    
    for U0, V0, W0 in inits:
        if time.time() > deadline:
            break
        U, V, W = U0.copy(), V0.copy(), W0.copy()
        
        for it in range(60):
            if time.time() > deadline:
                break
            
            # Update W
            for c in range(nwm):
                M = (U[:, None, :] * V[None, :, :]).reshape(nrm*nvm, r)
                t = T[:, :, c].ravel()
                W[c, :] = np.linalg.lstsq(M, t, rcond=None)[0]
            
            # Update U
            for a in range(nrm):
                M = (V[:, None, :] * W[None, :, :]).reshape(nvm*nwm, r)
                t = T[a, :, :].ravel()
                U[a, :] = np.linalg.lstsq(M, t, rcond=None)[0]
            
            # Update V
            for b in range(nvm):
                M = (U[:, None, :] * W[None, :, :]).reshape(nrm*nwm, r)
                t = T[:, b, :].ravel()
                V[b, :] = np.linalg.lstsq(M, t, rcond=None)[0]
            
            # Check residual
            if it % 10 == 0 or it == 59:
                res = _residual(U, V, W, T)
                if res < 1e-12:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _residual(U, V, W, T):
    res = 0.0
    for s in range(U.shape[1]):
        term = U[:, s][:, None, None] * V[:, s][None, :, None] * W[:, s][None, None, :]
        res += np.sum((term - T)**2)
    return res

def _strassen_42_arr(n, m, p):
    num_pairs = p // 2
    rank = num_pairs * 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    products = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)]),
    ]
    outc = [
        [((0,0),1),((1,1),1)],
        [((1,0),1),((1,1),-1)],
        [((0,1),1),((1,1),1)],
        [((0,0),1),((1,0),1)],
        [((0,0),-1),((0,1),1)],
        [((1,1),1)],
        [((0,0),1)],
    ]
    
    col = 0
    for t in range(num_pairs):
        bc = 2*t
        for s in range(7):
            for (ai,aj),cf in products[s][0]:
                U[ai*m+aj, col] = cf
            for (bj,bk),cf in products[s][1]:
                V[bj*p+bc+bk, col] = cf
            for (ci,ck),cf in outc[s]:
                W[(bc+ck)*n+ci, col] = cf
            col += 1
    return U, V, W

def _strassen_42(n, m, p):
    U, V, W = _strassen_42_arr(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
