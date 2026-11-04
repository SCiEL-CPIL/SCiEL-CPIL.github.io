import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Target tensor T[i, j, k] = 1
    T_target = np.ones((n, m, p))
    
    # Shapes:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Decomposition: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 20 to 23
    for rank in range(20, 24):
        if time.time() > deadline:
            break
            
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            def params_to_matrices(params):
                U_flat = params[:n*m*rank]
                V_flat = params[n*m*rank:2*n*m*rank]
                W_flat = params[2*n*m*rank:]
                return U_flat.reshape(n*m, rank), V_flat.reshape(m*p, rank), W_flat.reshape(p*n, rank)
                
            def objective(params):
                U, V, W = params_to_matrices(params)
                # Compute T_approx[i, j, k]
                # T_approx[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # We can loop over i, j, k for small dims
                T_approx = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_approx[i, j, k] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                diff = T_approx - T_target
                return np.sum(diff**2)
                
            # Gradient is complex to derive manually for this indexing, 
            # but we can use finite differences or just minimize without gradient (Nelder-Mead) 
            # if dims are small. 
            # Dims: U(6,20)=120, V(12,20)=240, W(8,20)=160. Total 520 vars.
            # Nelder-Mead might be slow.
            # Let's use scipy.optimize.minimize with 'L-BFGS-B' and numerical gradient.
            
            from scipy.optimize import minimize
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-9})
            
            if res.fun < 1e-6:
                U_opt, V_opt, W_opt = params_to_matrices(res.x)
                
                # Snap to half-integers
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify
                T_snap = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_snap[i, j, k] = np.sum(U_snap[i*m+j] * V_snap[j*p+k] * W_snap[k*n+i])
                            
                if np.allclose(T_snap, T_target, atol=1e-9):
                    # Found valid decomposition
                    best_rank = rank
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    break
        
        if best_rank == rank:
            break
            
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback schoolbook
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
