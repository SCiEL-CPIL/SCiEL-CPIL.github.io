import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135
    
    Nm = n * m  # 20
    Mp = m * p  # 30
    Pn = p * n  # 24
    
    T = np.zeros((Nm, Mp, Pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_norm = np.linalg.norm(T)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((Nm, rank))
        V = np.zeros((Mp, rank))
        W = np.zeros((Pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n*m*p
    
    def compute_error(U, V, W):
        T_pred = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.linalg.norm(T_pred - T) / T_norm
    
    found = False
    for target_rank in range(100, 120):
        if time.time() > deadline - 20:
            break
        for trial in range(2):
            if time.time() > deadline - 20:
                break
            np.random.seed(42 + target_rank * 100 + trial)
            U = np.random.randn(Nm, target_rank) * 0.5
            V = np.random.randn(Mp, target_rank) * 0.5
            W = np.random.randn(Pn, target_rank) * 0.5
            prev_err = np.inf
            for it in range(2000):
                if time.time() > deadline - 20:
                    break
                M_VW = np.einsum('br,cr->bcr', V, W).reshape(Mp*Pn, target_rank)
                T_flat = T.reshape(Nm, Mp*Pn)
                try:
                    U, *_ = np.linalg.lstsq(M_VW.T, T_flat.T, rcond=None)
                except:
                    break
                M_UW = np.einsum('ar,cr->acr', U, W).reshape(Nm*Pn, target_rank)
                T_fV = T.transpose(1,0,2).reshape(Mp, Nm*Pn)
                try:
                    V, *_ = np.linalg.lstsq(M_UW.T, T_fV.T, rcond=None)
                except:
                    break
                M_UV = np.einsum('ar,br->abr', U, V).reshape(Nm*Mp, target_rank)
                T_fW = T.transpose(2,0,1).reshape(Pn, Nm*Mp)
                try:
                    W, *_ = np.linalg.lstsq(M_UV.T, T_fW.T, rcond=None)
                except:
                    break
                err = compute_error(U, V, W)
                if err < 1e-10:
                    found = True
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
                if err > prev_err * 0.99:
                    break
                prev_err = err
            if found:
                break
        if found:
            break
    
    if found:
        Us, Vs, Ws = np.round(best_U*2)/2, np.round(best_V*2)/2, np.round(best_W*2)/2
        if compute_error(Us, Vs, Ws) < 1e-9:
            best_U, best_V, best_W = Us, Vs, Ws
        else:
            best_U, best_V, best_W = schoolbook()
            best_rank = n*m*p
    
    return {
        "U": [[float(x) for x in r] for r in best_U],
        "V": [[float(x) for x in r] for r in best_V],
        "W": [[float(x) for x in r] for r in best_W]
    }
