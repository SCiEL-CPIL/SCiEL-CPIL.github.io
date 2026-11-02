import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    T = np.ones((n, m, p))
    
    target_ranks = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        found = False
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n, m, R)
            V = np.random.randn(m, p, R)
            W = np.random.randn(p, n, R)
            
            for step in range(100):
                if time.time() > deadline:
                    break
                    
                # Solve for U
                # Equation: sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r]) = 1
                # Let C[i,j,k,r] = V[j,k,r] * W[k,i,r]
                C = V[:, :, None, :] * W[None, :, :, :] # Shape (m, p, n, R) -> wait
                # V is (m, p, R), W is (p, n, R)
                # We need V[j,k,r] * W[k,i,r].
                # Let's compute C[i,j,k,r]
                # C[i,j,k,r] = sum? No, it's a product for each r.
                # Actually, the term is U[i,j,r] * V[j,k,r] * W[k,i,r].
                # So for fixed i,j,k, the coefficient of U[i,j,r] is V[j,k,r] * W[k,i,r].
                
                # Vectorize U to shape (n*m, R) -> (n*m*R,)
                U_vec = U.reshape(-1)
                
                # Build matrix M for U
                # Rows correspond to (i,j,k)
                # Cols correspond to (i,j,r)
                # M[(i,j,k), (i,j,r)] = V[j,k,r] * W[k,i,r]
                M_U = np.zeros((n * m * p, n * m * R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row_idx = i * m * p + j * p + k
                            for r in range(R):
                                col_idx = (i * m + j) * R + r
                                M_U[row_idx, col_idx] = V[j, k, r] * W[k, i, r]
                
                # Solve M_U @ U_vec = T.flatten()
                U_vec, residuals, rank_u, s = np.linalg.lstsq(M_U, T.flatten(), rcond=None)
                U = U_vec.reshape(n, m, R)
                
                # Solve for V
                # Coefficient of V[j,k,r] is U[i,j,r] * W[k,i,r]
                V_vec = V.reshape(-1)
                M_V = np.zeros((n * m * p, m * p * R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row_idx = i * m * p + j * p + k
                            for r in range(R):
                                col_idx = (j * p + k) * R + r
                                M_V[row_idx, col_idx] = U[i, j, r] * W[k, i, r]
                
                V_vec, residuals, rank_v, s = np.linalg.lstsq(M_V, T.flatten(), rcond=None)
                V = V_vec.reshape(m, p, R)
                
                # Solve for W
                # Coefficient of W[k,i,r] is U[i,j,r] * V[j,k,r]
                W_vec = W.reshape(-1)
                M_W = np.zeros((n * m * p, p * n * R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row_idx = i * m * p + j * p + k
                            for r in range(R):
                                col_idx = (k * n + i) * R + r
                                M_W[row_idx, col_idx] = U[i, j, r] * V[j, k, r]
                
                W_vec, residuals, rank_w, s = np.linalg.lstsq(M_W, T.flatten(), rcond=None)
                W = W_vec.reshape(p, n, R)
                
                # Check error
                Est = np.einsum('ijr,jkr,kir->ijk', U, V, W)
                err_norm = np.linalg.norm(Est - T)
                
                if err_norm < 1e-9:
                    # Found solution
                    U_list = U.reshape(n * m, R).tolist()
                    V_list = V.reshape(m * p, R).tolist()
                    W_list = W.reshape(p * n, R).tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        
    # Fallback
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
