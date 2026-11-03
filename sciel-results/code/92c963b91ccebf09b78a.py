import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return U, V, W

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _snap_to_half_integer(x):
    return round(x * 2) / 2.0

def _snap_and_verify(U, V, W):
    U_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in U])
    V_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in V])
    W_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in W])
    if _verify(U_snap, V_snap, W_snap):
        return U_snap, V_snap, W_snap
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    U0, V0, W0 = _schoolbook()
    
    if not _verify(U0, V0, W0):
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = U0.shape[1]
    
    if time.time() < deadline - 10:
        target_rank = best_rank - 1
        np.random.seed(0)
        
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            for it in range(30):
                if time.time() > deadline - 2:
                    break
                
                for i in range(n):
                    for j in range(m):
                        M = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        try:
                            U[i*m+j, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        except:
                            pass
                
                for j in range(m):
                    for k in range(p):
                        M = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        try:
                            V[j*p+k, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                        except:
                            pass
                
                for k in range(p):
                    for i in range(n):
                        M = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        try:
                            W[k*n+i, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                        except:
                            pass
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err += (s - 1.0) ** 2
                if err < 1e-8:
                    break
            
            result = _snap_and_verify(U, V, W)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = result[0].shape[1]
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
