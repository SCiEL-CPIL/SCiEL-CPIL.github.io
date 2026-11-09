import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school), dtype=int)
    V_school = np.zeros((m*p, rank_school), dtype=int)
    W_school = np.zeros((p*n, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    # Target Tensor
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # Else 0
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0

    # Try to find rank 23
    target_rank = 23
    best_result = None
    
    # Optimization function
    def cost_func(params):
        U = params[0:n*m*target_rank].reshape(n*m, target_rank)
        V = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        
        # Compute tensor
        # T_recon[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
        # Using einsum
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        error = np.sum((T_recon - T_target)**2)
        return error

    # Try a few random restarts
    np.random.seed(42)
    for attempt in range(20):
        if time.time() > deadline - safety_margin:
            break
            
        # Initialize
        # Scale: since target is 1, and we want integer/half-integer, 
        # let's just optimize floats and round later.
        U_init = np.random.randn(n*m, target_rank) * 0.5
        V_init = np.random.randn(m*p, target_rank) * 0.5
        W_init = np.random.randn(p*n, target_rank) * 0.5
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = opt.minimize(cost_func, params_init, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e-6:
                # Extract solution
                U_sol = res.x[0:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = res.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                # Round to nearest 0.5
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                # Verify
                T_recon = np.einsum('ur,vr,wr->uvw', U_rounded, V_rounded, W_rounded)
                if np.allclose(T_recon, T_target, atol=1e-9):
                    best_result = (U_rounded, V_rounded, W_rounded)
                    break
        except Exception:
            pass

    if best_result:
        U, V, W = best_result
        # Convert to list of lists for interface
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
