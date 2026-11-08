import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
    rank_school = n * m * p
    U_school = np.zeros((N_U, rank_school))
    V_school = np.zeros((N_V, rank_school))
    W_school = np.zeros((N_W, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    T = np.zeros((N_U, N_V, N_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    timeout = 115.0

    for rank in range(best_rank - 1, 20, -1):
        if time.time() - start_time > timeout:
            break
            
        found = False
        for restart in range(10):
            if time.time() - start_time > timeout:
                break
                
            np.random.seed(42 + restart * 100 + rank)
            U = np.random.randn(N_U, rank)
            V = np.random.randn(N_V, rank)
            W = np.random.randn(N_W, rank)
            
            for it in range(50):
                if time.time() - start_time > timeout:
                    break
                    
                # Solve for U
                A = np.einsum('br,cr->bcr', V, W).reshape(N_V * N_W, rank)
                for a in range(N_U):
                    t = T[a, :, :].ravel()
                    try:
                        U[a, :], residuals, rank_solver, sv = np.linalg.lstsq(A, t, rcond=None)
                    except:
                        continue
                        
                # Solve for V
                A = np.einsum('ar,cr->acr', U, W).reshape(N_U * N_W, rank)
                for b in range(N_V):
                    t = T[:, b, :].ravel()
                    try:
                        V[b, :], residuals, rank_solver, sv = np.linalg.lstsq(A, t, rcond=None)
                    except:
                        continue
                        
                # Solve for W
                A = np.einsum('ar,br->abr', U, V).reshape(N_U * N_V, rank)
                for c in range(N_W):
                    t = T[:, :, c].ravel()
                    try:
                        W[c, :], residuals, rank_solver, sv = np.linalg.lstsq(A, t, rcond=None)
                    except:
                        continue
                        
            T_recon = np.tensordot(U, V, axes=(1, 1))
            T_recon = np.tensordot(T_recon, W, axes=(2, 1))
            error = np.linalg.norm(T_recon - T)
            
            if error < 1e-8:
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                T_recon_snapped = np.tensordot(U_snapped, V_snapped, axes=(1, 1))
                T_recon_snapped = np.tensordot(T_recon_snapped, W_snapped, axes=(2, 1))
                error_snapped = np.linalg.norm(T_recon_snapped - T)
                
                if error_snapped < 1e-8:
                    best_result = {
                        "U": U_snapped.tolist(),
                        "V": V_snapped.tolist(),
                        "W": W_snapped.tolist()
                    }
                    best_rank = rank
                    found = True
                    break
        
        if found:
            pass
        else:
            if time.time() - start_time > timeout - 10:
                break

    return best_result
