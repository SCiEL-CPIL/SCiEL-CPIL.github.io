import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 40  # Fallback
    best_U, best_V, best_W = None, None, None
    best_rank = target_rank
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Try ranks from lower bound up to 40
    # Lower bound is max(n*m, m*p, n*p) = 20
    for r in range(20, target_rank + 1):
        if time.time() > deadline:
            break
            
        # Try a few restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Coordinate descent
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # A_{k, alpha} = V[j*p+k, alpha] * W[k*n+i, alpha]
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b = np.ones(p)
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                            
                # Update V
                for j in range(m):
                    for k in range(p):
                        # A_{i, alpha} = U[i*m+j, alpha] * W[k*n+i, alpha]
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b = np.ones(n)
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                            
                # Update W
                for k in range(p):
                    for i in range(n):
                        # A_{j, alpha} = U[i*m+j, alpha] * V[j*p+k, alpha]
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b = np.ones(m)
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                            
                # Check error
                # T_{i,j,k} = sum_alpha U[i*m+j, alpha] V[j*p+k, alpha] W[k*n+i, alpha]
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            error += (val - 1.0) ** 2
                            
                if error < 1e-6:
                    # Found a solution
                    # Convert to list of lists
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    return {"U": U_list, "V": V_list, "W": W_list}
            
    # If no better rank found, return trivial rank 40
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
                
    return {"U": U, "V": V, "W": W}
