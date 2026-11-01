"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Returns a matrix multiplication tensor decomposition.
    Tries to find a rank-minimal exact decomposition for 2x3 @ 3x10.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # 5s safety margin before 120s kill
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 60
    
    # Try to find a lower rank decomposition
    # Strategy: The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # For 2x3x10, let's try ranks from 30 upward
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(6,30,20) = 30
    
    best_result = None
    
    # First, construct schoolbook as guaranteed fallback
    rank_sb = schoolbook_rank
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    def schoolbook_result():
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find a rank < 60 decomposition
    # Use a structured approach: since n=2, we can try to exploit symmetry
    
    # For each candidate rank r, set up the problem
    # Variables: U (6*r), V (30*r), W (20*r) -> total 56*r variables
    # Constraints: 60 equations (one for each (i,j,k))
    
    # Try a few candidate ranks
    candidate_ranks = [40, 45, 50, 55, 60]
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        if r >= schoolbook_rank:
            break
        
        # Initialize with random values
        np.random.seed(42)
        U_init = np.random.randn(n * m, r) * 0.1
        V_init = np.random.randn(m * p, r) * 0.1
        W_init = np.random.randn(p * n, r) * 0.1
        
        # Flatten
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective_and_grad(x):
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
            W = x[n*m*r+m*p*r:].reshape(p*n, r)
            
            # Compute the tensor from decomposition
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for rr in range(r):
                            val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                        error += (val - 1.0) ** 2
            return error
        
        # Try a few random initializations
        found = False
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + attempt)
            U_init = np.random.randn(n * m, r) * 0.5
            V_init = np.random.randn(m * p, r) * 0.5
            W_init = np.random.randn(p * n, r) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use scipy minimize
            try:
                result = minimize(objective_and_grad, x0, method='Nelder-Mead', 
                                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
                
                if result.fun < 1e-10:
                    U_opt = result.x[:n*m*r].reshape(n*m, r)
                    V_opt = result.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W_opt = result.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    # Verify
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for rr in range(r):
                                    val += U_opt[i*m+j, rr] * V_opt[j*p+k, rr] * W_opt[k*n+i, rr]
                                if abs(val - 1.0) > 1e-8:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                        found = True
                        break
                        
            except Exception:
                continue
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
    return schoolbook_result()
# EVOLVE-BLOCK-END
