import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Schoolbook decomposition
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
                
    T = np.ones((n, m, p))
    
    # Target rank
    target_rank = 20
    
    # Function to compute error
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        return err

    # Try to find rank 20 solution
    # We will use a simple local search on the half-integer grid
    # or scipy optimization
    
    # Since the space is discrete (half-integers), we can't just use gradient descent
    # directly on the continuous space and expect to land on half-integers easily.
    # However, the optimal solution for these tensors often has small integer coefficients.
    # We can use scipy.optimize.least_squares on the continuous variables,
    # and then check if the result is close to half-integers.
    
    from scipy.optimize import least_squares
    
    def residuals(x):
        U = x[:n*m*target_rank].reshape(n*m, target_rank)
        V = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res.append(val - 1.0)
        return np.array(res)
    
    best_solution = None
    best_rank = rank_school
    
    rng = np.random.default_rng(42)
    
    # Try a few random initializations
    # The coefficients are likely small.
    # Initialize with small random values.
    
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        # Random initialization in [-1, 1]
        x0 = rng.uniform(-1, 1, size=(n*m + m*p + p*n) * target_rank)
        
        try:
            res = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if res.success or res.cost < 1e-8:
                U = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W = res.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                # Check if close to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify exactness with rounded values
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                            err += (val - 1.0)**2
                            
                if err < 1e-9:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        except Exception as e:
            pass

    # If no rank 20 found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
