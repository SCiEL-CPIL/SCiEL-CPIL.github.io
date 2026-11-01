import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_decomposition(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=1e-9)
    
    def try_rank(r, num_restarts=20):
        if time.time() > deadline:
            return None
        
        best_err = float('inf')
        best_params = None
        
        # Number of variables: r * (nm + mp + pn)
        size = r * (n*m + m*p + p*n)
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            params = np.random.randn(size) * 0.5
            
            def objective(params):
                U = params[:r*n*m].reshape(n*m, r)
                V = params[r*n*m:r*(n*m+m*p)].reshape(m*p, r)
                W = params[r*(n*m+m*p):].reshape(p*n, r)
                
                T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
                err = np.sum((T_approx - target)**2)
                return err
            
            def gradient(params):
                U = params[:r*n*m].reshape(n*m, r)
                V = params[r*n*m:r*(n*m+m*p)].reshape(m*p, r)
                W = params[r*(n*m+m*p):].reshape(p*n, r)
                
                T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
                diff = T_approx - target
                
                # Gradient w.r.t U: sum_{b,c} diff[a,b,c] * V[b,k] * W[c,k]
                dU = np.einsum('abc,bk,ck->ak', diff, V, W)
                # Gradient w.r.t V: sum_{a,c} diff[a,b,c] * U[a,k] * W[c,k]
                dV = np.einsum('abc,ak,ck->bk', diff, U, W)
                # Gradient w.r.t W: sum_{a,b} diff[a,b,c] * U[a,k] * V[b,k]
                dW = np.einsum('abc,ak,bk->ck', diff, U, V)
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return grad
            
            # Use L-BFGS-B with gradient
            res = minimize(objective, params, jac=gradient, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500})
            
            if time.time() > deadline:
                break
            
            err = res.fun
            if err < best_err:
                best_err = err
                best_params = res.x.copy()
                
                if err < 1e-9:
                    U = res.x[:r*n*m].reshape(n*m, r)
                    V = res.x[r*n*m:r*(n*m+m*p)].reshape(m*p, r)
                    W = res.x[r*(n*m+m*p):].reshape(p*n, r)
                    
                    # Try rounding to nearest 0.5
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    if check_decomposition(U_rounded, V_rounded, W_rounded):
                        return U_rounded, V_rounded, W_rounded
                    elif check_decomposition(U, V, W):
                        return U, V, W
        
        return None
    
    # Try rank 20 first
    res = try_rank(20, num_restarts=15)
    if res is not None:
        U, V, W = res
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 21 if time permits
    if time.time() < deadline - 30:
        res = try_rank(21, num_restarts=10)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 24
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
# EVOLVE-BLOCK-END
