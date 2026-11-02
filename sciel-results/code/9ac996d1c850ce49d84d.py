import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # T[i,k,j] = 1 for all i in range(n), k in range(p), j in range(m)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Try increasing rank until we find a solution
    start_time = time.time()
    deadline = start_time + 120
    
    # Start with a reasonable guess and go up
    # Lower bound: at least max(n*p, m*p, p*n) / something... 
    # Actually, the rank must be at least the "slice rank" or similar
    # Let's just try from r=1 upward
    
    best_result = None
    
    for r in range(1, 25):
        if time.time() > deadline - 5:
            break
            
        # Set up the system: 24 equations, 3*r unknowns
        # Use scipy to solve the nonlinear system
        from scipy.optimize import least_squares
        
        def residuals(x):
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W = x[n*m*r + m*p*r:].reshape(p*n, r)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        res.append(val - 1.0)
            return res
        
        # Try multiple random initializations
        found = False
        max_iters = 50
        
        for trial in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(trial * 1000 + r)
            x0 = np.random.randn(n*m*r + m*p*r + p*n*r) * 0.5
            # Add some structure: try making some entries 1
            x0[:r] = 1.0  # First row of U
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=10000)
                if result.success or result.cost < 1e-10:
                    # Verify
                    x = result.x
                    U = x[:n*m*r].reshape(n*m, r)
                    V = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                    W = x[n*m*r + m*p*r:].reshape(p*n, r)
                    
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for c in range(r):
                                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                                max_err = max(max_err, abs(val - 1.0))
                    
                    if max_err < 1e-8:
                        # Round to nearest half-integer and verify
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        
                        max_err2 = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = 0.0
                                    for c in range(r):
                                        val += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                                    max_err2 = max(max_err2, abs(val - 1.0))
                        
                        if max_err2 < 1e-10:
                            # Found a valid half-integer solution
                            best_result = (r, U_r.tolist(), V_r.tolist(), W_r.tolist())
                            found = True
                            break
            except Exception:
                pass
        
        if found:
            break
    
    if best_result is None:
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
    else:
        r, U, V, W = best_result
        return {"U": U, "V": V, "W": W}
