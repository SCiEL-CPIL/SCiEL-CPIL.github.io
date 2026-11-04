import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 125

    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    rank_sb = n*m*p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    T_W = T.reshape(n*m*m*p, p*n)
    T_V = T.transpose(0,2,1).reshape(n*m*p*n, m*p)
    T_U = T.transpose(1,2,0).reshape(m*p*p*n, n*m)

    for rank in range(35, rank_sb):
        if time.time() > deadline - 20:
            break
        success = False
        for seed in range(50):
            if time.time() > deadline - 15:
                break
            np.random.seed(seed)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)

            for it in range(150):
                if time.time() > deadline - 10:
                    break
                UV = np.einsum('ar,br->abr', U, V).reshape(n*m*m*p, rank)
                W = np.linalg.lstsq(UV, T_W, rcond=None)[0]
                UW = np.einsum('ar,cr->acr', U, W).reshape(n*m*p*n, rank)
                V = np.linalg.lstsq(UW, T_V, rcond=None)[0]
                VW = np.einsum('br,cr->bcr', V, W).reshape(m*p*p*n, rank)
                U = np.linalg.lstsq(VW, T_U, rcond=None)[0]

                if it % 10 == 9:
                    recon = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.max(np.abs(recon - T))
                    if err < 1e-8:
                        success = True
                        break

            if success:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break

        if success:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
