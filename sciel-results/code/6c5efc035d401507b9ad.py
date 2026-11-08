import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower rank
    # We iterate downwards from best_rank - 1
    # We limit the number of trials to fit in time
    
    def verify(U, V, W):
        # Check if U, V, W are integers or half-integers and decompose exactly
        # Convert to list for checking
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # Check types
        for row in U_list:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        for row in V_list:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        for row in W_list:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check decomposition
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Target is 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(U.shape[1]):
                        val += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def search_rank(r):
        if r >= best_rank:
            return None
        
        # Use random restarts
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Optimization
            # We minimize sum (T - sum U V W)^2
            # This is non-convex.
            # We can use gradient descent or scipy.optimize.
            # Given small dimensions, simple GD might work if initialized well.
            # But let's try a few steps of a more robust method or just check if random luck finds it? Unlikely.
            
            # A better approach: use the fact that for small r, we can solve for one matrix given the others.
            # If we fix U and V, the equations for W are linear?
            # Sum_c U_ic V_jc W_kc = 1.
            # This is not linear in W because of the sum over c?
            # Actually, for fixed i,j, sum_c (U_ic V_jc) W_kc = 1.
            # This is a linear system in W_k for each k?
            # No, W_kc is the variable.
            # Let A_c = U_ic V_jc. Then sum_c A_c W_kc = 1.
            # This must hold for all i,j.
            # So for each k, we have a system of equations for W_kc?
            # No, W_kc is shared across i,j.
            
            # Let's stick to numerical optimization on the flattened vector.
            # Flatten U, V, W into a single vector x.
            # Loss L(x) = sum_{i,j,k} (1 - sum_c U_ic V_jc W_kc)^2
            
            # This is computationally heavy if done naively.
            # n=2, m=3, p=5.
            # U size: 6 x r. V size: 15 x r. W size: 10 x r.
            # Total params: (6+15+10)r = 31r.
            # For r=29, ~900 params.
            # Equations: 2*3*5 = 30.
            # Overdetermined.
            
            # We can use scipy.optimize.least_squares
            import scipy.optimize as opt
            
            def loss_func(x):
                U_flat = x[:n*m*r]
                V_flat = x[n*m*r : n*m*r + m*p*r]
                W_flat = x[n*m*r + m*p*r : ]
                
                U_mat = U_flat.reshape(n*m, r)
                V_mat = V_flat.reshape(m*p, r)
                W_mat = W_flat.reshape(p*n, r)
                
                residuals = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                            residuals.append(1.0 - val)
                return np.array(residuals)
                
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = opt.least_squares(loss_func, x0, method='lm', max_nfev=1000)
                if res.success or res.cost < 1e-10:
                    U_sol = res.x[:n*m*r].reshape(n*m, r)
                    V_sol = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W_sol = res.x[n*m*r + m*p*r : ].reshape(p*n, r)
                    
                    if verify(U_sol, V_sol, W_sol):
                        return (U_sol, V_sol, W_sol, r)
            except Exception:
                pass
                
        return None

    # Try ranks from 29 down to 20 (arbitrary cutoff, lower bounds suggest 15)
    for r in range(29, 14, -1):
        if time.time() > deadline:
            break
        result = search_rank(r)
        if result:
            best_U, best_V, best_W, best_rank = result
            break
            
    # Return in required format
    # Convert numpy arrays to lists of lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
