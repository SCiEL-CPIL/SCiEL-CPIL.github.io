import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for 120s limit
    
    n, m, p = DIMENSIONS
    nm, mp, np_ = n * m, m * p, p * n
    
    def residual(x, R):
        # x is flattened [U (nm*R), V (mp*R), W (np_*R)]
        U = x[:nm*R].reshape(nm, R)
        V = x[nm*R:nm*R+mp*R].reshape(mp, R)
        W = x[nm*R+mp*R:].reshape(np_, R)
        
        # Residuals: for each i, k, j1, j2
        # sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] - delta[j1, j2]
        res = np.zeros((n, p, m, m))
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        # Compute sum
                        # U[i*m+j1, :] * V[j2*p+k, :] * W[k*n+i, :]
                        s = np.sum(U[i*m+j1, :] * V[j2*p+k, :] * W[k*n+i, :])
                        target = 1.0 if j1 == j2 else 0.0
                        res[i, k, j1, j2] = s - target
        return res.flatten()

    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((np_, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    # Try to find lower rank
    for R in range(rank_school - 1, 80, -1):
        if time.time() > deadline - 10:
            break
            
        # Initialize with first R schoolbook terms
        x0 = np.concatenate([
            U_school[:, :R].flatten(),
            V_school[:, :R].flatten(),
            W_school[:, :R].flatten()
        ])
        
        try:
            # Use least_squares
            # Note: This might be slow or fail to converge for large R
            # We use a small max_nfev to keep it fast
            res = least_squares(residual, x0, args=(R,), max_nfev=1000, xtol=1e-8, ftol=1e-8)
            
            if res.success and res.cost < 1e-6:
                # Check if we can round to 0.5
                x_opt = res.x
                U_opt = x_opt[:nm*R].reshape(nm, R)
                V_opt = x_opt[nm*R:nm*R+mp*R].reshape(mp, R)
                W_opt = x_opt[nm*R+mp*R:].reshape(np_, R)
                
                # Round to nearest 0.5
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Verify rounded solution
                res_check = residual(np.concatenate([
                    U_rounded.flatten(),
                    V_rounded.flatten(),
                    W_rounded.flatten()
                ]), R)
                
                if np.linalg.norm(res_check) < 1e-4:
                    best_result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    best_rank = R
                    break # Found a better one, stop searching lower ranks immediately? 
                    # Or continue to find even lower? 
                    # Given time, break.
        
        except Exception as e:
            pass

    return best_result
# EVOLVE-BLOCK-END
