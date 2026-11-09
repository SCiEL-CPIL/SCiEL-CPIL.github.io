import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin before 150s limit
    
    # Target tensor for 2x3 * 3x9
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (27, rank)
    # W shape: (p*n, rank) = (18, rank)
    # Tensor: T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    
    target_indices = np.array(target_indices)
    target_a = target_indices[:, 0]
    target_b = target_indices[:, 1]
    target_c = target_indices[:, 2]
    N = len(target_indices)
    
    best_rank = n * m * p  # 54
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 53 down to 40
    # We prioritize finding *any* rank < 54 to beat the seed.
    for r in range(53, 39, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        max_restarts = 100 # More restarts allowed due to faster eval
        
        while max_restarts > 0 and not found:
            if time.time() > deadline - 5:
                break
            
            # Random initialization with integers {-1, 0, 1}
            # This increases the chance of finding an exact integer solution
            np.random.seed(r * 1000 + max_restarts)
            U = np.random.choice([-1, 0, 1], size=(n * m, r))
            V = np.random.choice([-1, 0, 1], size=(m * p, r))
            W = np.random.choice([-1, 0, 1], size=(p * n, r))
            
            # Convert to float for optimization
            U = U.astype(float)
            V = V.astype(float)
            W = W.astype(float)
            
            def objective(params):
                U_p = params[:n*m*r].reshape(n*m, r)
                V_p = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_p = params[n*m*r + m*p*r:].reshape(p*n, r)
                
                U_idx = U_p[target_a]
                V_idx = V_p[target_b]
                W_idx = W_p[target_c]
                
                T_vals = np.sum(U_idx * V_idx * W_idx, axis=1)
                diff = T_vals - 1.0
                return np.sum(diff * diff)
            
            def grad(params):
                U_p = params[:n*m*r].reshape(n*m, r)
                V_p = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_p = params[n*m*r + m*p*r:].reshape(p*n, r)
                
                U_idx = U_p[target_a]
                V_idx = V_p[target_b]
                W_idx = W_p[target_c]
                
                T_vals = np.sum(U_idx * V_idx * W_idx, axis=1)
                diff = T_vals - 1.0 # Shape (N,)
                
                # Gradient calculation
                # dL/dU[a,t] = sum_{idx: a_i=a} 2 * diff_i * V[b_i,t] * W[c_i,t]
                grad_U = np.zeros_like(U_p)
                grad_V = np.zeros_like(V_p)
                grad_W = np.zeros_like(W_p)
                
                # Vectorized gradient accumulation
                # contrib_U has shape (N, r)
                contrib_U = 2.0 * diff[:, None] * V_idx * W_idx
                contrib_V = 2.0 * diff[:, None] * U_idx * W_idx
                contrib_W = 2.0 * diff[:, None] * U_idx * V_idx
                
                np.add.at(grad_U, target_a, contrib_U)
                np.add.at(grad_V, target_b, contrib_V)
                np.add.at(grad_W, target_c, contrib_W)
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            p0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                from scipy.optimize import minimize
                # Use L-BFGS-B with tight constraints to find a local min quickly
                res = minimize(objective, p0, jac=grad, method='L-BFGS-B', 
                              options={'maxiter': 100, 'gtol': 1e-5, 'maxfun': 200, 'ftol': 1e-8})
                
                if res.fun < 1e-6:
                    # Try to round to half-integers
                    U_r = np.round(res.x[:n*m*r].reshape(n*m, r) * 2) / 2
                    V_r = np.round(res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r) * 2) / 2
                    W_r = np.round(res.x[n*m*r + m*p*r:].reshape(p*n, r) * 2) / 2
                    
                    # Verify
                    U_idx = U_r[target_a]
                    V_idx = V_r[target_b]
                    W_idx = W_r[target_c]
                    T_vals = np.sum(U_idx * V_idx * W_idx, axis=1)
                    err = np.sum((T_vals - 1.0) ** 2)
                    
                    if err < 1e-9:
                        best_rank = r
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        found = True
                        break
            except Exception:
                pass
            
            max_restarts -= 1
        
        if found:
            break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fall back to schoolbook
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
