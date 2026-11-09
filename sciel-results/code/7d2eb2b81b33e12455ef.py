import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    def error_fn(params, r):
        U = params[0: r*n*m].reshape(r, n*m)
        V = params[r*n*m : r*n*m + r*m*p].reshape(r, m*p)
        W = params[r*n*m + r*m*p : ].reshape(r, p*n)
        
        U_3d = U.T.reshape(n, m, r)
        V_3d = V.T.reshape(m, p, r)
        W_3d = W.T.reshape(p, n, r)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U_3d[i, j, c] * V_3d[j, k, c] * W_3d[k, i, c]
                    err += (val - 1.0)**2
        return err

    def get_matrices(params, r):
        U = params[0: r*n*m].reshape(r, n*m).T
        V = params[r*n*m : r*n*m + r*m*p].reshape(r, m*p).T
        W = params[r*n*m + r*m*p : ].reshape(r, p*n).T
        return U, V, W

    def is_half_integer(x, tol=1e-4):
        diff = np.abs(x - np.round(x * 2) / 2.0)
        return np.all(diff < tol)
        
    def verify(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try to find a solution with rank < 18
    for r in range(15, 9, -1):
        if time.time() > deadline:
            break
            
        for attempt in range(30):
            if time.time() > deadline:
                break
                
            np.random.seed(int(time.time()*1000) + attempt + r)
            
            # Initialize with small random values
            U_init = np.random.randn(n*m, r) * 0.5
            V_init = np.random.randn(m*p, r) * 0.5
            W_init = np.random.randn(p*n, r) * 0.5
            
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(error_fn, params, args=(r,), method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-6:
                    U, V, W = get_matrices(res.x, r)
                    
                    if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                        U_snapped = np.round(U * 2) / 2.0
                        V_snapped = np.round(V * 2) / 2.0
                        W_snapped = np.round(W * 2) / 2.0
                        
                        if verify(U_snapped, V_snapped, W_snapped, r):
                            return {
                                "U": U_snapped.tolist(),
                                "V": V_snapped.tolist(),
                                "W": W_snapped.tolist()
                            }
            except Exception:
                pass

    # Fallback to schoolbook
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
