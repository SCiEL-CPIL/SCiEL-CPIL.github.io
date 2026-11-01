"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 18
    pn = p * n  # 12
    
    # Target ranks to try, from low to high
    # Schoolbook is 36. Try to find lower.
    best_result = None
    best_rank = nm * mp // m  # 36
    
    # Try various ranks
    for target_rank in range(20, 37):
        r = target_rank
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        # Define the residual: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
        def residual(params):
            U_flat = params[:nm*r].reshape(nm, r)
            V_flat = params[nm*r:nm*r+mp*r].reshape(mp, r)
            W_flat = params[nm*r+mp*r:].reshape(pn, r)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c]
                        res.append(val - 1.0)
            return np.array(res)
        
        # Use least squares to solve
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Try to solve using nonlinear least squares
        try:
            from scipy.optimize import least_squares
            result = least_squares(residual, x0, method='lm', max_nfev=10000)
            if result.success or np.max(np.abs(result.fun)) < 1e-10:
                U_sol = result.x[:nm*r].reshape(nm, r)
                V_sol = result.x[nm*r:nm*r+mp*r].reshape(mp, r)
                W_sol = result.x[nm*r+mp*r:].reshape(pn, r)
                
                # Check if we can convert to half-integers
                # Try rounding to nearest half-integer
                U_half = np.round(U_sol * 2) / 2
                V_half = np.round(V_sol * 2) / 2
                W_half = np.round(W_sol * 2) / 2
                
                # Verify
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U_half[i*m+j, c] * V_half[j*p+k, c] * W_half[k*n+i, c]
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-9:
                    # Convert to list format
                    U_list = [[float(x) for x in row] for row in U_half]
                    V_list = [[float(x) for x in row] for row in V_half]
                    W_list = [[float(x) for x in row] for row in W_half]
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass
    
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
# EVOLVE-BLOCK-END
