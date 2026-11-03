"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, n*p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_a = T.reshape(n*m, -1)
    T_b = T.transpose(1,0,2).reshape(m*p, -1)
    T_c = T.transpose(2,0,1).reshape(n*p, -1)
    
    deadline = time.time() + 125
    
    for r in range(25, 73):
        if time.time() > deadline:
            break
        found = False
        for seed in range(8):
            if time.time() > deadline:
                break
            rng = np.random.RandomState(seed)
            U = rng.randn(n*m, r)
            V = rng.randn(m*p, r)
            W = rng.randn(n*p, r)
            
            for it in range(500):
                # Update U
                VW = (V[:, :, None] * W[None, :, :]).reshape(m*p*n*p, r).T
                U, *_ = np.linalg.lstsq(VW, T_a.T, rcond=None)
                U = U.T
                # Update V
                UW = (U[:, :, None] * W[None, :, :]).reshape(n*m*n*p, r).T
                V, *_ = np.linalg.lstsq(UW, T_b.T, rcond=None)
                V = V.T
                # Update W
                UV = (U[:, :, None] * V[None, :, :]).reshape(n*m*m*p, r).T
                W, *_ = np.linalg.lstsq(UV, T_c.T, rcond=None)
                W = W.T
                
                if it % 50 == 0:
                    recon = np.einsum('ar,br,cr->abc', U, V, W)
                    if np.max(np.abs(recon - T)) < 1e-12:
                        break
            
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(recon - T))
            if err < 1e-10:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                recon2 = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.max(np.abs(recon2 - T)) < 1e-10:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                # Try quarter integers
                U_r = np.round(U * 4) / 4
                V_r = np.round(V * 4) / 4
                W_r = np.round(W * 4) / 4
                recon2 = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.max(np.abs(recon2 - T)) < 1e-10:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                found = True
                break
        
        if found:
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
