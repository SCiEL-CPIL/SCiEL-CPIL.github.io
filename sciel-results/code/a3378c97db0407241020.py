import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p
    
    def objective_and_grad(x, r):
        U_flat = x[:r*n*m]
        V_flat = x[r*n*m:r*(n*m + m*p)]
        W_flat = x[r*(n*m + m*p):]
        
        U = U_flat.reshape(n*m, r)
        V = V_flat.reshape(m*p, r)
        W = W_flat.reshape(p*n, r)
        
        # Compute S_ijk
        # S[i, j, k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        S = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    S[i, j, k] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    
        res = S - 1.0
        cost = np.sum(res * res)
        
        # Gradient
        # dE/dU[i*m+j, t] = 2 * sum_k res[i, j, k] * V[j*p+k, t] * W[k*n+i, t]
        # dE/dV[j*p+k, t] = 2 * sum_i res[i, j, k] * U[i*m+j, t] * W[k*n+i, t]
        # dE/dW[k*n+i, t] = 2 * sum_j res[i, j, k] * U[i*m+j, t] * V[j*p+k, t]
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r_val = res[i, j, k]
                    if r_val == 0: continue
                    # Accumulate to U
                    grad_U[i*m+j] += 2 * r_val * V[j*p+k] * W[k*n+i]
                    # Accumulate to V
                    grad_V[j*p+k] += 2 * r_val * U[i*m+j] * W[k*n+i]
                    # Accumulate to W
                    grad_W[k*n+i] += 2 * r_val * U[i*m+j] * V[j*p+k]
                    
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return cost, grad

    def try_rank(r, max_restarts=5):
        if time.time() > deadline - 10:
            return None
            
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            x0 = np.random.uniform(-1.0, 1.0, size=r*(n*m + m*p + p*n))
            
            try:
                res = minimize(objective_and_grad, x0, args=(r,), method='BFGS', jac=True, options={'maxiter': 100, 'maxls': 20})
            except Exception:
                continue
                
            if res.fun < 1e-10:
                U_flat = res.x[:r*n*m]
                V_flat = res.x[r*n*m:r*(n*m + m*p)]
                W_flat = res.x[r*(n*m + m*p):]
                
                U_snapped = np.round(U_flat * 2) / 2
                V_snapped = np.round(V_flat * 2) / 2
                W_snapped = np.round(W_flat * 2) / 2
                
                U_s = U_snapped.reshape(n*m, r)
                V_s = V_snapped.reshape(m*p, r)
                W_s = W_snapped.reshape(p*n, r)
                
                is_exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U_s[i*m+j] * V_s[j*p+k] * W_s[k*n+i])
                            if not np.isclose(s, 1.0, atol=1e-9):
                                is_exact = False
                                break
                        if not is_exact:
                            break
                    if not is_exact:
                        break
                        
                if is_exact:
                    return {
                        "U": U_s.tolist(),
                        "V": V_s.tolist(),
                        "W": W_s.tolist()
                    }
        
        return None

    for r in range(20, trivial_rank + 1):
        if time.time() > deadline - 15:
            break
            
        result = try_rank(r, max_restarts=5)
        if result is not None:
            return result
            
    U = [[0] * trivial_rank for _ in range(n * m)]
    V = [[0] * trivial_rank for _ in range(m * p)]
    W = [[0] * trivial_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
