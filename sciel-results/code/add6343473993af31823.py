import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    def compute_residual(U, V, W):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err
    
    def als_decompose(U_init, V_init, W_init, r, max_iter=1000, tol=1e-12):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        max_err = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A_w = np.zeros((m, r))
                    for j in range(m):
                        A_w[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    W[k*n+i, :], _, _, _ = np.linalg.lstsq(A_w, np.ones(m), rcond=None)
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A_u = np.zeros((p, r))
                    for k in range(p):
                        A_u[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    U[i*m+j, :], _, _, _ = np.linalg.lstsq(A_u, np.ones(p), rcond=None)
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A_v = np.zeros((n, r))
                    for i in range(n):
                        A_v[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    V[j*p+k, :], _, _, _ = np.linalg.lstsq(A_v, np.ones(n), rcond=None)
            
            # Check residual
            max_err = compute_residual(U, V, W)
            
            if max_err < tol:
                break
        
        return U, V, W, max_err
    
    # Try rank 17 with multiple random initializations
    for attempt in range(200):
        if time.time() > deadline - 20:
            break
        
        np.random.seed(attempt * 31 + 13)
        r = 17
        U0 = np.random.randn(n*m, r) * 0.5
        V0 = np.random.randn(m*p, r) * 0.5
        W0 = np.random.randn(p*n, r) * 0.5
        
        U, V, W, err = als_decompose(U0, V0, W0, r, max_iter=300)
        
        if err < 1e-8:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If close, try to polish with more iterations
        if err < 0.1:
            U, V, W, err2 = als_decompose(U, V, W, r, max_iter=1000)
            if err2 < 1e-8:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook rank 18
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
