import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    num_U = n * m  # 9
    num_V = m * p  # 15
    num_W = p * n  # 15
    
    # Schoolbook solution (Rank 45)
    best_rank = 45
    best_U = np.zeros((num_U, best_rank))
    best_V = np.zeros((num_V, best_rank))
    best_W = np.zeros((num_W, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1

    def get_schoolbook():
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }

    # Vectorized Objective Function
    def objective(x, R):
        U = x[:num_U*R].reshape(num_U, R)
        V = x[num_U*R:num_U*R + num_V*R].reshape(num_V, R)
        W = x[num_U*R + num_V*R:].reshape(num_W, R)
        
        # Compute tensor T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # Using einsum for efficiency
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Error is sum of squares of (T - 1)
        error = np.sum((T - 1.0)**2)
        return error

    def verify_solution(U, V, W):
        # Check if the decomposition is exact (within floating point tolerance)
        # and entries are close to half-integers
        R = U.shape[1]
        
        # Round to nearest half-integer
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Check exactness with rounded values
        T = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
        
        # Check if T is all 1s
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        
        # Check if entries are actually half-integers (they should be by construction of rounding)
        # But we also want to ensure they are not just close, but effectively exact in the decomposition
        # The problem statement says "exactly decompose".
        # If rounded values give 1.0, it's a valid half-integer solution.
        
        return True

    # Search for lower ranks
    # Try ranks 44 down to 41
    for R in range(44, 40, -1):
        if time.time() > deadline - 10:
            break
            
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(restart * 100 + R)
            
            # Initialization: Random small values
            # Sometimes starting near schoolbook is better, but we can't fit 45 terms in R < 45 slots easily.
            # We can try to initialize with a "compressed" schoolbook?
            # For now, random init is the standard approach for these searches.
            
            x0 = np.random.randn(num_U*R + num_V*R + num_W*R) * 0.5
            
            try:
                # Use L-BFGS-B for speed
                res = minimize(objective, x0, args=(R,), method='L-BFGS-B', 
                               options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-8:
                    # Extract solution
                    U_sol = res.x[:num_U*R].reshape(num_U, R)
                    V_sol = res.x[num_U*R:num_U*R + num_V*R].reshape(num_V, R)
                    W_sol = res.x[num_U*R + num_V*R:].reshape(num_W, R)
                    
                    if verify_solution(U_sol, V_sol, W_sol):
                        return {
                            "U": (np.round(U_sol * 2) / 2).tolist(),
                            "V": (np.round(V_sol * 2) / 2).tolist(),
                            "W": (np.round(W_sol * 2) / 2).tolist()
                        }
            except Exception:
                pass

    return get_schoolbook()
