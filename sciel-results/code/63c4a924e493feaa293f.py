"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20-second safety margin before 150s kill
    
    # Build target tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i consistent
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ALS for increasing ranks
    for rank in [25, 30, 35, 40, 45, 50]:
        if time.time() > deadline:
            break
        for restart in range(3):
            if time.time() > deadline:
                break
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            for it in range(50):
                if time.time() > deadline:
                    break
                # ALS: update U
                M_U = (V[:, :, None] * W[None, :, :]).reshape(rank, m*p * p*n)
                T_mat = T.reshape(n*m, m*p * p*n)
                U = np.linalg.lstsq(M_U.T, T_mat.T, rcond=None)[0].T
                # ALS: update V
                M_V = (U[:, :, None] * W[None, :, :]).reshape(rank, n*m * p*n)
                T_mat_v = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
                V = np.linalg.lstsq(M_V.T, T_mat_v.T, rcond=None)[0].T
                # ALS: update W
                M_W = (U[:, :, None] * V[None, :, :]).reshape(rank, n*m * m*p)
                T_mat_w = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
                W = np.linalg.lstsq(M_W.T, T_mat_w.T, rcond=None)[0].T
                # Check error periodically
                if it % 10 == 9:
                    recon = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.linalg.norm(recon - T)
                    if err < 1e-12:
                        # Try snapping to half-integers
                        Us = np.round(U * 2) / 2
                        Vs = np.round(V * 2) / 2
                        Ws = np.round(W * 2) / 2
                        if np.linalg.norm(np.einsum('ar,br,cr->abc', Us, Vs, Ws) - T) < 1e-10:
                            return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
