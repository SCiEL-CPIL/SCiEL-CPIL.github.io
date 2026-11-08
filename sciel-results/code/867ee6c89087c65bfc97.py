"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # safety margin
    
    # Schoolbook as fallback
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
    best_U, best_V, best_W = U_school, V_school, W_school
    
    def check_validity(U, V, W, tol=1e-6):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank decompositions
    for R in range(5, rank_school):  # Try ranks from 5 up to schoolbook-1
        if time.time() > deadline - 30:
            break
        
        # Random initialization
        best_obj = float('inf')
        found = False
        
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
            
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Scale to get closer to 1
            # The sum of products should be 1, so let's normalize
            # Each term contributes, so scale each matrix by 1/R^(1/3)
            scale = 1.0 / (R ** (1.0/3.0))
            U *= scale
            V *= scale
            W *= scale
            
            def objective(x):
                if time.time() > deadline - 10:
                    return float('inf')
                U_r = x[:n*m*R].reshape(n*m, R)
                V_r = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_r = x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Compute constraint violations
                violations = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i])
                            violations += (val - 1.0) ** 2
                return violations
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = minimize(objective, x0, method='BFGS', maxiter=100, 
                                  options={'gtol': 1e-8, 'maxfun': 200})
                if result.fun < best_obj:
                    best_obj = result.fun
                    U_best = result.x[:n*m*R].reshape(n*m, R)
                    V_best = result.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W_best = result.x[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    if best_obj < 1e-8:
                        # Check validity
                        if check_validity(U_best, V_best, W_best, tol=1e-5):
                            found = True
                            best_rank = R
                            best_U, best_V, best_W = U_best.copy(), V_best.copy(), W_best.copy()
                            break
            except Exception:
                pass
        
        if found:
            break
    
    # Convert to lists with integer/half-integer entries if possible
    # For now, return as floats (the task says integer/half-integer, but let's see)
    # Actually, the task says "integer/half-integer entries" - this is a constraint!
    # Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries"
    
    # This is a significant constraint. Let me try to find integer/half-integer solutions.
    # For the schoolbook, entries are 0 or 1, which are integers.
    # For lower rank, we need to find integer/half-integer entries.
    
    # Let me try a different approach: search over small integer/half-integer values.
    # This is a combinatorial problem.
    
    # For now, return the best we found. If it's not integer/half-integer, we may need to round.
    # But rounding may break the decomposition.
    
    # Let's return the schoolbook if we didn't find a better valid integer/half-integer solution.
    # For the numerical solution, let's try to round to nearest half-integer and check.
    
    if best_rank < rank_school:
        # Try to round to half-integers
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        if check_validity(U_rounded, V_rounded, W_rounded, tol=1e-4):
            best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
            best_rank = R
        else:
            # Keep the schoolbook
            best_U, best_V, best_W = U_school, V_school, W_school
            best_rank = rank_school
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
