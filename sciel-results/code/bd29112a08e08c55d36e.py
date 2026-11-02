import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 4, 7]

def _error(params, n, m, p, r):
    U = params[0: n*m*r].reshape(n*m, r)
    V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
    W = params[n*m*r + m*p*r :].reshape(p*n, r)
    
    # Compute the tensor contraction
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is a 3D tensor.
    # We can compute this efficiently using einsum or loops.
    # Given small dimensions, loops are fine.
    
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over c
                s = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(r):
                    s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                err += (s - 1.0) ** 2
    return err

def _grad(params, n, m, p, r):
    U = params[0: n*m*r].reshape(n*m, r)
    V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
    W = params[n*m*r + m*p*r :].reshape(p*n, r)
    
    # Gradient of 0.5 * sum (s-1)^2
    # d/dU[u,c] = sum_{j,k} (s_{ijk} - 1) * V[j*p+k, c] * W[k*n+i, c]
    # Note: s_{ijk} depends on U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    gU = np.zeros_like(U)
    gV = np.zeros_like(V)
    gW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = 0.0
                for c in range(r):
                    s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                err_term = s - 1.0
                for c in range(r):
                    gU[u_idx, c] += err_term * V[v_idx, c] * W[w_idx, c]
                    gV[v_idx, c] += err_term * U[u_idx, c] * W[w_idx, c]
                    gW[w_idx, c] += err_term * U[u_idx, c] * V[v_idx, c]
    
    return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100 # 20s buffer
    
    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find a better rank
    # Start from lower bound
    lower_bound = max(n*m, m*p, p*n)
    
    # We try a few ranks. 
    # Ranks to try: 28, 30, 35, 40, 45, 50, 55
    # If we find one, we return it.
    # We prioritize lower ranks.
    
    ranks_to_try = list(range(lower_bound, best_rank))
    # Shuffle or order? Order by rank ascending.
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Multiple random restarts for each rank
        # Fewer restarts for lower ranks if they fail quickly?
        # Let's do 5 restarts for each rank.
        for restart in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(restart + r) # Reproducible-ish
            # Initialize
            # Scale matters. 
            # If U, V, W are small, products are small.
            # Let's try initializing with values around 1.
            U_init = np.random.randn(n*m, r)
            V_init = np.random.randn(m*p, r)
            W_init = np.random.randn(p*n, r)
            
            # Normalize?
            # If we scale U by a, V by b, W by c, the product scales by abc.
            # We want the sum to be 1.
            # Let's just use random init.
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimization
            # Use Nelder-Mead or BFGS?
            # BFGS is good if gradient is available.
            # We have gradient.
            
            # Limit iterations
            max_iter = 50
            res = minimize(_error, x0, method='BFGS', jac=_grad, 
                           args=(n, m, p, r), 
                           options={'maxiter': max_iter, 'gtol': 1e-4})
            
            if res.fun < 1e-6:
                # Found a solution
                U_opt = res.x[0: n*m*r].reshape(n*m, r)
                V_opt = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_opt = res.x[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Verify
                # Check error again
                if _error(res.x, n, m, p, r) < 1e-6:
                    best_rank = r
                    best_U = U_opt
                    best_V = V_opt
                    best_W = W_opt
                    # Since we are iterating r ascending, the first one found is the best for this search order.
                    # But maybe a lower rank exists? We are going up from lower bound.
                    # So if we find r=28, we return it.
                    # If we find r=30, we continue to see if we can find 29? 
                    # No, we are iterating 28, 29, 30...
                    # So if we find a solution at r, it's the lowest found so far.
                    # We can break out of the restart loop, but should we continue to lower r? 
                    # We are already at the lowest r in the loop.
                    # So we can return immediately?
                    # Yes, because we want the *minimal* rank.
                    # If we find rank 28, it's likely optimal or close.
                    # If we find rank 30, we should check 29? 
                    # The loop goes 28, 29, 30...
                    # So if we fail at 28, we try 29. If we succeed at 29, we return 29.
                    # This is correct.
                    
                    # Convert to lists
                    return {
                        "U": best_U.tolist(),
                        "V": best_V.tolist(),
                        "W": best_W.tolist()
                    }
            
    # If no better rank found, return schoolbook
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
