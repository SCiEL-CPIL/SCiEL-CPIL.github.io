import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    rank = 48
    
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds margin
    
    np.random.seed(42)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_res = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(500):
        if time.time() > deadline:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                try:
                    AAT = A @ A.T
                    u = np.linalg.solve(AAT, b)
                    U[i * m + j, :] = A.T @ u
                except np.linalg.LinAlgError:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                try:
                    AAT = A @ A.T
                    v = np.linalg.solve(AAT, b)
                    V[j * p + k, :] = A.T @ v
                except np.linalg.LinAlgError:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                try:
                    AAT = A @ A.T
                    w = np.linalg.solve(AAT, b)
                    W[k * n + i, :] = A.T @ w
                except np.linalg.LinAlgError:
                    pass
        
        # Check convergence
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    res += (val - 1.0) ** 2
        
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if res < 1e-8:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
