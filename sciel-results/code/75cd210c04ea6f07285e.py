import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Dimensions of the factor matrices
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36
    
    # Construct the target tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit

    # Try ranks from 143 down to 100
    # Schoolbook is 144.
    for R in range(143, 99, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random restarts for this rank
        for restart in range(3):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(42 + restart * 1000 + R)
            U = np.random.randn(dim_u, R) * 0.1
            V = np.random.randn(dim_v, R) * 0.1
            W = np.random.randn(dim_w, R) * 0.1
            
            # Regularization for stability
            lam = 1e-8
            I_R = np.eye(R) * lam
            
            for iter in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U
                VtV = V.T @ V
                WtW = W.T @ W
                G_U = VtV * WtW + I_R
                R_U = np.einsum('vc,wc,uvw->uc', V, W, T, optimize=True)
                try:
                    U = np.linalg.solve(G_U, R_U.T).T
                except np.linalg.LinAlgError:
                    U = np.linalg.lstsq(G_U, R_U.T, rcond=None)[0].T
                
                # Update V
                UtU = U.T @ U
                G_V = UtU * WtW + I_R
                R_V = np.einsum('uc,wc,uvw->vc', U, W, T, optimize=True)
                try:
                    V = np.linalg.solve(G_V, R_V.T).T
                except np.linalg.LinAlgError:
                    V = np.linalg.lstsq(G_V, R_V.T, rcond=None)[0].T
                
                # Update W
                VtV = V.T @ V
                G_W = UtU * VtV + I_R
                R_W = np.einsum('uc,vc,uvw->wc', U, V, T, optimize=True)
                try:
                    W = np.linalg.solve(G_W, R_W.T).T
                except np.linalg.LinAlgError:
                    W = np.linalg.lstsq(G_W, R_W.T, rcond=None)[0].T
                
                # Check error every few iterations
                if iter % 5 == 0 or iter == 49:
                    T_recon = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
                    error = np.linalg.norm(T_recon - T)
                    if error < 1e-6:
                        # Try to round to half-integers
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        
                        T_recon_r = np.einsum('uc,vc,wc->uvw', U_rounded, V_rounded, W_rounded, optimize=True)
                        error_r = np.linalg.norm(T_recon_r - T)
                        
                        if error_r < 1e-9:
                            return {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }

    # If no better rank found, return schoolbook
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
