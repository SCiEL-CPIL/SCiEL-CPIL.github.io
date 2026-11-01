import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 50
    
    # Define target tensor support
    idx_a = []
    idx_b = []
    idx_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a.append(i*m + j)
                idx_b.append(j*p + k)
                idx_c.append(k*n + i)
                
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    idx_c = np.array(idx_c)
    
    # Initialize with schoolbook decomposition of rank 54, then project to rank 50?
    # Or just random.
    np.random.seed(42)
    
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(x):
        U_flat, V_flat, W_flat = x[:n*m*target_rank], x[n*m*target_rank:2*n*m*target_rank], x[2*n*m*target_rank:]
        U_curr = U_flat.reshape(n*m, target_rank)
        V_curr = V_flat.reshape(m*p, target_rank)
        W_curr = W_flat.reshape(p*n, target_rank)
        
        U_sub = U_curr[idx_a, :]
        V_sub = V_curr[idx_b, :]
        W_sub = W_curr[idx_c, :]
        
        T_curr_vals = np.sum(U_sub * V_sub * W_sub, axis=1)
        
        err = np.sum((T_curr_vals - 1)**2)
        
        return err
    
    start_time = time.time()
    result = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 2000, 'maxfun': 20000})
    
    elapsed = time.time() - start_time
    
    U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
    V_opt = result.x[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
    W_opt = result.x[2*n*m*target_rank:].reshape(p*n, target_rank)
    
    U_list = U_opt.tolist()
    V_list = V_opt.tolist()
    W_list = W_opt.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
