import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 2, 2, 12
    
    # Strategy: Try to find a rank-30 decomposition using 3 blocks of rank-10 2x2x4.
    # If it fails, fall back to rank-42 Strassen.
    
    # Define sparsity pattern for 10 terms of 2x2x4
    # U: 4 entries, V: 8 entries, W: 8 entries per term
    # We assume a dense pattern for simplicity in the solver, but sparse in reality?
    # Let's assume dense for the solver to find any solution, then check sparsity?
    # No, rank 10 implies 10 rank-1 tensors. They can be dense.
    
    # Target tensor T of shape (4, 8, 8) for 2x2x4 block
    # T[i*m+j, j*p4+k, k*n+i] = 1
    # i in 0..1, j in 0..1, k in 0..3
    # U index: i*m+j
    # V index: j*p4+k
    # W index: k*n+i
    
    T_target = np.zeros((4, 8, 8))
    for i in range(2):
        for j in range(2):
            for k in range(4):
                u_idx = i * 2 + j
                v_idx = j * 4 + k
                w_idx = k * 2 + i
                T_target[u_idx, v_idx, w_idx] = 1.0
    
    # Variables: 10 terms * (4 + 8 + 8) = 200 variables
    # Flatten U, V, W for all terms
    num_terms = 10
    dim_u = 4
    dim_v = 8
    dim_w = 8
    total_vars = num_terms * (dim_u + dim_v + dim_w)
    
    def objective(params):
        # Reshape params into U, V, W arrays
        # U: (num_terms, dim_u)
        # V: (num_terms, dim_v)
        # W: (num_terms, dim_w)
        
        U = params[0:num_terms*dim_u].reshape(num_terms, dim_u)
        V = params[num_terms*dim_u:num_terms*dim_u + num_terms*dim_v].reshape(num_terms, dim_v)
        W = params[num_terms*dim_u + num_terms*dim_v:].reshape(num_terms, dim_w)
        
        # Compute tensor sum
        # T_approx[i, j, k] = sum_r U[r, i] * V[r, j] * W[r, k]
        # Use einsum
        T_approx = np.einsum('ri,rj,rs->ijs', U, V, W)
        
        # Error
        err = T_approx - T_target
        return np.sum(err**2)
    
    # Initial guess: Random
    np.random.seed(42)
    x0 = np.random.randn(total_vars) * 0.1
    
    # Optimize
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success and res.fun < 1e-6:
            # Extract solution
            U_sol = res.x[0:num_terms*dim_u].reshape(num_terms, dim_u)
            V_sol = res.x[num_terms*dim_u:num_terms*dim_u + num_terms*dim_v].reshape(num_terms, dim_v)
            W_sol = res.x[num_terms*dim_u + num_terms*dim_v:].reshape(num_terms, dim_w)
            
            # Build global matrices
            rank = 30
            U_global = np.zeros((4, rank))
            V_global = np.zeros((8, rank))
            W_global = np.zeros((24, rank)) # p*n = 12*2 = 24
            
            for b in range(3):
                base_rank = b * 10
                base_col = b * 4
                
                for t in range(10):
                    r = base_rank + t
                    
                    # U
                    for i in range(2):
                        for j in range(2):
                            u_idx = i * 2 + j
                            U_global[u_idx, r] = U_sol[t, u_idx]
                            
                    # V
                    for j in range(2):
                        for k_loc in range(4):
                            v_idx = j * 4 + k_loc
                            k_glob = base_col + k_loc
                            V_global[j * 12 + k_glob, r] = V_sol[t, v_idx]
                            
                    # W
                    for k_loc in range(4):
                        for i in range(2):
                            w_idx = k_loc * 2 + i
                            k_glob = base_col + k_loc
                            W_global[k_glob * 2 + i, r] = W_sol[t, w_idx]
                            
            # Verify
            U_arr = U_global
            V_arr = V_global
            W_arr = W_global
            
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                        if abs(val - 1.0) > max_err:
                            max_err = abs(val - 1.0)
                            
            if max_err < 1e-5:
                return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
                
    except Exception as e:
        pass
    
    # Fallback: Rank 42 Strassen
    rank = 42
    U = np.zeros((4, rank))
    V = np.zeros((8, rank))
    W = np.zeros((24, rank))
    
    # Strassen terms for 2x2x2
    terms_strassen = [
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(0,1): 1, (1,1): -1}},
        {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(1,0): 1, (1,1): 1}},
        {'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (0,1): 1}},
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (1,0): 1}},
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}},
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}}
    ]
    
    for b in range(6):
        base_rank = b * 7
        base_col = b * 2
        for t_idx in range(7):
            r = base_rank + t_idx
            term = terms_strassen[t_idx]
            for (i, j), val in term['U'].items():
                U[i * 2 + j, r] = val
            for (j, k_loc), val in term['V'].items():
                k_glob = base_col + k_loc
                V[j * 12 + k_glob, r] = val
            for (k_loc, i), val in term['W'].items():
                k_glob = base_col + k_loc
                W[k_glob * 2 + i, r] = val
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
