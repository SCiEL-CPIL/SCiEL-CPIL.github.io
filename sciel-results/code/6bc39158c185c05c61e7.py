import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    lower_bound = m * (n + p - 1)
    
    start_time = time.time()
    time_limit = 100
    
    # Schoolbook solution
    U_sb = np.zeros((dim_U, rank_schoolbook))
    V_sb = np.zeros((dim_V, rank_schoolbook))
    W_sb = np.zeros((dim_W, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
                
    for r in range(lower_bound, min(rank_schoolbook, 50) + 1):
        if time.time() - start_time > time_limit:
            break
            
        for restart in range(10):
            if time.time() - start_time > time_limit:
                break
                
            U = np.random.randn(dim_U, r)
            V = np.random.randn(dim_V, r)
            W = np.random.randn(dim_W, r)
            
            for _ in range(30):
                # Update U
                # For each i, j, solve (A^T A) x = A^T b
                # A is p x r, A[k, c] = sum? No, A[k, c] = V[j, k, c] * W[k, i, c]
                # V is (m, p, r), W is (p, n, r)
                V_reshaped = V.reshape(m, p, r)
                W_reshaped = W.reshape(p, n, r)
                
                for i in range(n):
                    for j in range(m):
                        # A[k, c] = V[j, k, c] * W[k, i, c]
                        A = V_reshaped[j] * W_reshaped[:, i, :] # Shape (p, r)
                        # b is ones of length p
                        b = np.ones(p)
                        
                        # Solve A x = b
                        # x = np.linalg.lstsq(A, b, rcond=None)[0]
                        # Use normal equations for speed if r is small
                        AtA = A.T @ A
                        Atb = A.T @ b
                        try:
                            x = np.linalg.solve(AtA, Atb)
                        except np.linalg.LinAlgError:
                            x = np.linalg.lstsq(AtA, Atb, rcond=None)[0]
                        
                        U[i*m + j, :] = x
                        
                # Update V
                # For each j, k, solve (A^T A) x = A^T b
                # A[i, c] = U[i, j, c] * W[k, i, c]
                U_reshaped = U.reshape(n, m, r)
                
                for j in range(m):
                    for k in range(p):
                        # A[i, c] = U[i, j, c] * W[k, i, c]
                        # U[i, j, c] is U_reshaped[i, j, :]
                        # W[k, i, c] is W_reshaped[k, i, :]
                        A = U_reshaped[:, j, :] * W_reshaped[k, :, :] # Shape (n, r)
                        b = np.ones(n)
                        
                        AtA = A.T @ A
                        Atb = A.T @ b
                        try:
                            x = np.linalg.solve(AtA, Atb)
                        except np.linalg.LinAlgError:
                            x = np.linalg.lstsq(AtA, Atb, rcond=None)[0]
                        
                        V[j*p + k, :] = x
                        
                # Update W
                # For each k, i, solve (A^T A) x = A^T b
                # A[j, c] = U[i, j, c] * V[j, k, c]
                
                for k in range(p):
                    for i in range(n):
                        # A[j, c] = U[i, j, c] * V[j, k, c]
                        # U[i, j, c] is U_reshaped[i, j, :]
                        # V[j, k, c] is V_reshaped[j, k, :]
                        A = U_reshaped[i, :, :] * V_reshaped[:, k, :] # Shape (m, r)
                        b = np.ones(m)
                        
                        AtA = A.T @ A
                        Atb = A.T @ b
                        try:
                            x = np.linalg.solve(AtA, Atb)
                        except np.linalg.LinAlgError:
                            x = np.linalg.lstsq(AtA, Atb, rcond=None)[0]
                        
                        W[k*n + i, :] = x
            
            # Check residual
            U_reshaped = U.reshape(n, m, r)
            V_reshaped = V.reshape(m, p, r)
            W_reshaped = W.reshape(p, n, r)
            
            T_approx = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
            residual = np.sum((T_approx - 1.0)**2)
            
            if residual < 1e-6:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
