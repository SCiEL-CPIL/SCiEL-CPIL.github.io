import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # n=3 rows of A, m=4 inner, p=6 cols of B
    
    # Precompute M_full once (the target tensor)
    M_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M_full[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_result = None
    best_rank = None
    rng = np.random.RandomState(42)
    
    # Search ranks from 50 to 72
    for rank in range(50, 73):
        if time.time() > deadline - 20:
            break
        
        # Multiple restarts with different seeds
        for restart in range(10):
            if time.time() > deadline - 15:
                break
            
            U = rng.randn(n*m, rank)
            V = rng.randn(m*p, rank)
            W = rng.randn(p*n, rank)
            
            converged = False
            for iteration in range(500):
                if time.time() > deadline - 5:
                    break
                
                U_r = U.reshape(n, m, rank)
                V_r = V.reshape(m, p, rank)
                W_r = W.reshape(p, n, rank)
                
                # Update U
                N_U = np.einsum('jkc,kic->ijc', V_r, W_r)
                VtV = V.T @ V
                WtW = W.T @ W
                G_U = VtV * WtW
                
                try:
                    N_U_flat = N_U.reshape(n*m, rank)
                    U_new = np.linalg.solve(G_U, N_U_flat.T).T
                except np.linalg.LinAlgError:
                    G_U_reg = G_U + np.eye(rank) * 1e-6
                    try:
                        U_new = np.linalg.solve(G_U_reg, N_U_flat.T).T
                    except:
                        break
                U = U_new
                
                # Update V
                U_r = U.reshape(n, m, rank)
                N_V = np.einsum('ijc,kic->jk c', U_r, W_r)
                UtU = U.T @ U
                WtW = W.T @ W
                G_V = UtU * WtW
                
                try:
                    N_V_flat = N_V.reshape(m*p, rank)
                    V_new = np.linalg.solve(G_V, N_V_flat.T).T
                except np.linalg.LinAlgError:
                    G_V_reg = G_V + np.eye(rank) * 1e-6
                    try:
                        V_new = np.linalg.solve(G_V_reg, N_V_flat.T).T
                    except:
                        break
                V = V_new
                
                # Update W
                U_r = U.reshape(n, m, rank)
                V_r = V.reshape(m, p, rank)
                N_W = np.einsum('ijc,jkc->ikc', U_r, V_r)
                UtU = U.T @ U
                VtV = V.T @ V
                G_W = UtU * VtV
                
                try:
                    N_W_flat = N_W.reshape(p*n, rank)
                    W_new = np.linalg.solve(G_W, N_W_flat.T).T
                except np.linalg.LinAlgError:
                    G_W_reg = G_W + np.eye(rank) * 1e-6
                    try:
                        W_new = np.linalg.solve(G_W_reg, N_W_flat.T).T
                    except:
                        break
                W = W_new
                
                # Check convergence (every 5 iterations to save time)
                if iteration % 5 == 0 or iteration == 499:
                    U_r = U.reshape(n, m, rank)
                    V_r = V.reshape(m, p, rank)
                    W_r = W.reshape(p, n, rank)
                    Val = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                    err1 = np.max(np.abs(Val - 1.0))
                    
                    if err1 < 1e-6:
                        # Quick full check
                        T_full = np.zeros((n*m, m*p, p*n))
                        for c in range(rank):
                            T_full += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                        err2 = np.max(np.abs(T_full - M_full))
                        residual = max(err1, err2)
                        if residual < 1e-8:
                            converged = True
                            break
            
            if converged:
                # Final polish pass with tighter tolerance
                for polish_iter in range(50):
                    if time.time() > deadline - 2:
                        break
                    
                    U_r = U.reshape(n, m, rank)
                    V_r = V.reshape(m, p, rank)
                    W_r = W.reshape(p, n, rank)
                    
                    N_U = np.einsum('jkc,kic->ijc', V_r, W_r)
                    VtV = V.T @ V
                    WtW = W.T @ W
                    G_U = VtV * WtW
                    try:
                        U = np.linalg.solve(G_U, N_U.reshape(n*m, rank).T).T
                    except:
                        break
                    
                    U_r = U.reshape(n, m, rank)
                    N_V = np.einsum('ijc,kic->jk c', U_r, W_r)
                    UtU = U.T @ U
                    WtW = W.T @ W
                    G_V = UtU * WtW
                    try:
                        V = np.linalg.solve(G_V, N_V.reshape(m*p, rank).T).T
                    except:
                        break
                    
                    U_r = U.reshape(n, m, rank)
                    V_r = V.reshape(m, p, rank)
                    N_W = np.einsum('ijc,jkc->ikc', U_r, V_r)
                    UtU = U.T @ U
                    VtV = V.T @ V
                    G_W = UtU * VtV
                    try:
                        W = np.linalg.solve(G_W, N_W.reshape(p*n, rank).T).T
                    except:
                        break
                
                best_rank = rank
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
