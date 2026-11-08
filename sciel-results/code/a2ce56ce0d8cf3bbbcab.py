import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = start_time + 110
    
    for r in range(best_rank - 1, 20, -1):
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(30):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            prev_err = np.inf
            for it in range(50):
                if time.time() > deadline:
                    break
                    
                U_3d = U.reshape(n, m, r)
                V_3d = V.reshape(m, p, r)
                W_3d = W.reshape(p, n, r)
                
                # Solve for W
                M = np.zeros((n * m * p, p * n * r))
                for c in range(r):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                row_idx = i * m * p + j * p + k
                                col_idx = (k * n + i) * r + c
                                M[row_idx, col_idx] = U_3d[i, j, c] * V_3d[j, k, c]
                                
                T_flat = T.flatten()
                w_sol, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
                if np.any(np.isnan(w_sol)):
                    break
                W = w_sol.reshape(p, n, r).transpose(1, 0, 2).reshape(p * n, r)
                
                # Solve for V
                M = np.zeros((n * m * p, m * p * r))
                for c in range(r):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                row_idx = i * m * p + j * p + k
                                col_idx = (j * p + k) * r + c
                                M[row_idx, col_idx] = U_3d[i, j, c] * W_3d[k, i, c]
                                
                v_sol, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
                if np.any(np.isnan(v_sol)):
                    break
                V = v_sol.reshape(m, p, r).reshape(m * p, r)
                
                # Solve for U
                M = np.zeros((n * m * p, n * m * r))
                for c in range(r):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                row_idx = i * m * p + j * p + k
                                col_idx = (i * m + j) * r + c
                                M[row_idx, col_idx] = V_3d[j, k, c] * W_3d[k, i, c]
                                
                u_sol, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
                if np.any(np.isnan(u_sol)):
                    break
                U = u_sol.reshape(n, m, r).reshape(n * m, r)
                
                # Check error
                U_3d = U.reshape(n, m, r)
                V_3d = V.reshape(m, p, r)
                W_3d = W.reshape(p, n, r)
                T_approx = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
                err = np.linalg.norm(T - T_approx)
                
                if err < 1e-8:
                    # Round
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify
                    U_3d = U_rounded.reshape(n, m, r)
                    V_3d = V_rounded.reshape(m, p, r)
                    W_3d = W_rounded.reshape(p, n, r)
                    T_approx = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
                    
                    if np.allclose(T, T_approx, atol=1e-9):
                        best_rank = r
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        found = True
                        break
                elif err < prev_err * 0.99:
                    prev_err = err
                else:
                    break
                    
            if found:
                break
                
        if found:
            continue
        else:
            pass

    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
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
