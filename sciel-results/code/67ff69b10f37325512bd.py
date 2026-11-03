import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (12, rank)
    # W: (p*n, rank) = (12, rank)
    
    # Define the residual function for scipy.optimize.least_squares
    def residual(params, rank):
        U = params[0: n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank: n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank: n*m*rank + m*p*rank + p*n*rank].reshape(p*n, rank)
        
        # Residuals: for each (i,j,k), sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] - 1
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U[i*m + j, :]
                    v = V[j*p + k, :]
                    w = W[k*n + i, :]
                    res[idx] = np.dot(u, v * w) - 1.0
                    idx += 1
        return res

    best_rank = n * m * p
    best_solution = None
    
    # We will try ranks from a lower bound up to the schoolbook rank.
    # Lower bound for 3x3x4 is at least 15? Let's start from 15.
    # If we find a solution at rank R, we can stop if we want the minimal, 
    # but the problem asks for rank-minimal. So we should check all lower ranks first.
    
    # To speed up, we can limit the search to ranks where a solution is likely.
    # For 3x3x3, rank 23 is known. For 3x3x4, it might be around 24-36.
    # Let's try ranks from 15 to 36.
    
    # Use a set of initializations. 
    # Strategy: Random Half-Integers in {-1, -0.5, 0, 0.5, 1}
    # This biases the search towards the valid output space.
    
    seeds = [0, 1, 42, 123, 999]
    
    for rank in range(15, n * m * p + 1):
        found = False
        for seed in seeds:
            np.random.seed(seed + rank)
            # Initialize with half-integers
            vals = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m + m*p + p*n) * rank)
            x0 = vals
            
            try:
                sol = opt.least_squares(residual, x0, args=(rank,), method='lm', max_nfev=2000)
                if sol.success or sol.cost < 1e-8:
                    # Extract and round
                    U = sol.x[0: n*m*rank].reshape(n*m, rank)
                    V = sol.x[n*m*rank: n*m*rank + m*p*rank].reshape(m*p, rank)
                    W = sol.x[n*m*rank + m*p*rank: n*m*rank + m*p*rank + p*n*rank].reshape(p*n, rank)
                    
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify exactness
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_rank = rank
                        best_solution = (U_r, V_r, W_r)
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    if best_solution is not None:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
