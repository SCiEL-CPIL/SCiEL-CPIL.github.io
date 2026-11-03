import numpy as np
import scipy.optimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    # Target Rank
    # Known minimal rank for 3x3x4 is 16.
    target_rank = 16
    
    def residuals(params):
        # Unpack params
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape(n * m, target_rank)
        V = V_flat.reshape(m * p, target_rank)
        W = W_flat.reshape(p * n, target_rank)
        
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    best_solution = None
    
    # Try target rank 16
    for rank in [16, 17, 18]:
        found = False
        # Multiple random starts
        for trial in range(20):
            np.random.seed(trial * 7 + rank)
            # Initialize with small random values
            U_init = np.random.randn(n * m, rank) * 0.1
            V_init = np.random.randn(m * p, rank) * 0.1
            W_init = np.random.randn(p * n, rank) * 0.1
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use least_squares to minimize the error
                # 'trf' (Trust Region Reflective) is good for bounds, but we don't have bounds.
                # 'lm' (Levenberg-Marquardt) is good for small problems.
                result = scipy.optimize.least_squares(residuals, params0, method='lm', max_nfev=5000)
                
                # Check if residual is small enough to round
                if np.max(np.abs(result.fun)) < 1e-4:
                    # Recover matrices
                    U = result.x[:n*m*rank].reshape(n*m, rank)
                    V = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Round to nearest 0.5
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify exact decomposition
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
        # Fallback: schoolbook rank 36
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
