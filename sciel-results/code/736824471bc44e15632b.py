"""Find rank-minimal tensor decomposition for 2x5 x 5x6 matrix multiplication."""
DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    import numpy as np
    from scipy.optimize import minimize
    import time
    
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Build the target tensor T[i,j,k] = 1 if i+k==j else 0
    T = np.zeros((n, m, p))
    for i in range(n):
        for k in range(p):
            j = i + k
            if 0 <= j < m:
                T[i, j, k] = 1.0
    
    # Flatten: T_flat[i*m + j, k*n + i] ... no, let's use the proper indexing
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Number of non-zero entries in T
    nz = np.sum(T)
    
    # Schoolbook rank
    rank_schoolbook = int(nz)
    
    # Try to find a decomposition with rank < rank_schoolbook
    # Use a randomized approach: initialize with random low-rank factors and optimize
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from nz-1 down to some minimum
    # For each rank, try a few random initializations
    
    min_rank_to_try = max(1, nz - 15)  # Try ranks from nz-1 down to nz-15
    
    for rank in range(min(best_rank, nz), min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations for this rank
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(42 + rank * 100 + attempt)
            U0 = np.random.randn(n * m, rank) * 0.1
            V0 = np.random.randn(m * p, rank) * 0.1
            W0 = np.random.randn(p * n, rank) * 0.1
            
            # Define the objective: sum of squared residuals
            def objective(params):
                U = params[:n*m*rank].reshape(n*m, rank)
                V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            target = T[i, j, k]
                            residual += (val - target) ** 2
                return residual
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Use L-BFGS-B
            result = minimize(objective, params0, method='L-BFGS-B', 
                            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.fun < 1e-8:
                # Found a valid decomposition!
                U = result.x[:n*m*rank].reshape(n*m, rank)
                V = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Verify
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            err = abs(val - T[i, j, k])
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-6:
                    best_rank = rank
                    best_U = U
                    best_V = V
                    best_W = W
                    break  # Move to next (lower) rank
        
        if best_rank < rank:
            break  # Found a better rank, continue searching lower
    
    # If no better rank found, use schoolbook
    if best_U is None:
        rank = rank_schoolbook
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[i, j, k] != 0:
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
    else:
        U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
