"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-5):
    """Verify tensor decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 110  # safety margin
    
    # Schoolbook baseline: rank = n*m*p = 100
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    
    # Try to find lower rank decompositions via numerical optimization
    from scipy.optimize import minimize
    
    for target_rank in [95, 90, 85, 80, 75, 70]:
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            
            # Initialize from schoolbook (first target_rank columns) + perturbation
            np.random.seed(attempt * 7919 + target_rank)
            U_init = U_sb[:, :target_rank].copy() + np.random.uniform(-0.05, 0.05, (n*m, target_rank))
            V_init = V_sb[:, :target_rank].copy() + np.random.uniform(-0.05, 0.05, (m*p, target_rank))
            W_init = W_sb[:, :target_rank].copy() + np.random.uniform(-0.05, 0.05, (p*n, target_rank))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def error_fn(x):
                U_x = x[:n*m*target_rank].reshape(n*m, target_rank)
                V_x = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_x = x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U_x[i*m+j, c] * V_x[j*p+k, c] * W_x[k*n+i, c]
                            err += (s - 1.0) ** 2
                return err
            
            def grad_fn(x):
                U_x = x[:n*m*target_rank].reshape(n*m, target_rank)
                V_x = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_x = x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                dU = np.zeros_like(U_x)
                dV = np.zeros_like(V_x)
                dW = np.zeros_like(W_x)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U_x[i*m+j, c] * V_x[j*p+k, c] * W_x[k*n+i, c]
                            ds = 2.0 * (s - 1.0)
                            for c in range(target_rank):
                                dU[i*m+j, c] += ds * V_x[j*p+k, c] * W_x[k*n+i, c]
                                dV[j*p+k, c] += ds * U_x[i*m+j, c] * W_x[k*n+i, c]
                                dW[k*n+i, c] += ds * U_x[i*m+j, c] * V_x[j*p+k, c]
                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
            try:
                res = minimize(error_fn, x0, jac=grad_fn, method='L-BFGS-B',
                               options={'maxiter': 20, 'maxfun': 50, 'ftol': 1e-12})
                if res.fun < 1e-8:
                    U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank).tolist()
                    V_opt = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank).tolist()
                    W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank).tolist()
                    if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-4):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_opt, V_opt, W_opt
                        found = True
                        break
            except Exception:
                continue
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
