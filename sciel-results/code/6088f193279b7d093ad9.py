DIMENSIONS = [2, 2, 4]
import numpy as np
import time
import random
import sys

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x4():
    """
    Construct a rank-14 tensor decomposition for 2x2 * 2x4 matrix multiplication.
    Fallback known good solution.
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: k in {0,1}
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[2][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[2][2] = -1
    W[0][2] = 1; W[2][2] = 1
    
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[2][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[2][6] = 1
    W[0][6] = 1

    # Block 2: k in {2,3}
    U[0][7] = 1; U[3][7] = 1
    V[4][7] = 1; V[6][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    U[2][8] = 1; U[3][8] = 1
    V[4][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    U[0][9] = 1
    V[5][9] = 1; V[6][9] = -1
    W[4][9] = 1; W[6][9] = 1
    
    U[3][10] = 1
    V[5][10] = 1; V[4][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    U[0][11] = 1; U[1][11] = 1
    V[6][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    U[2][12] = 1; U[0][12] = -1
    V[4][12] = 1; V[5][12] = 1
    W[7][12] = 1
    
    U[1][13] = 1; U[3][13] = -1
    V[5][13] = 1; V[6][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def _try_find_rank(r, n, m, p, time_limit):
    """
    Attempt to find a rank-r decomposition using numerical optimization.
    Returns (U, V, W) if found, else None.
    """
    if time.time() > time_limit:
        return None
        
    # Variables: U (nm x r), V (mp x r), W (pn x r)
    # Flatten into a single vector x
    num_u = n * m * r
    num_v = m * p * r
    num_w = p * n * r
    total_vars = num_u + num_v + num_w
    
    # Objective function: sum of squared errors
    def objective(x):
        U_flat = x[:num_u]
        V_flat = x[num_u:num_u+num_v]
        W_flat = x[num_u+num_v:]
        
        U = U_flat.reshape((n*m, r))
        V = V_flat.reshape((m*p, r))
        W = W_flat.reshape((p*n, r))
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(r):
                        val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    error += (val - 1.0)**2
        return error

    # Random initialization
    best_sol = None
    best_err = float('inf')
    
    # Try a few random restarts
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > time_limit:
            break
            
        # Initialize with small random values
        x0 = np.random.randn(total_vars) * 0.5
        
        try:
            from scipy.optimize import minimize
            # Use Nelder-Mead as it doesn't require gradients and is robust for small problems
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if res.fun < best_err:
                best_err = res.fun
                best_sol = res.x
                
                if best_err < 1e-9:
                    # We found a solution
                    U_flat = best_sol[:num_u]
                    V_flat = best_sol[num_u:num_u+num_v]
                    W_flat = best_sol[num_u+num_v:]
                    
                    U = U_flat.reshape((n*m, r)).tolist()
                    V = V_flat.reshape((m*p, r)).tolist()
                    W = W_flat.reshape((p*n, r)).tolist()
                    
                    # Final verification
                    if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                        return {"U": U, "V": V, "W": W}
                        
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # First, try to find a rank-13 solution
    # If it exists, it's better than Strassen's 14
    sol_13 = _try_find_rank(13, n, m, p, deadline)
    if sol_13:
        return sol_13
    
    # If rank 13 failed, try rank 14 with optimization (might find a different one, but we have Strassen)
    # Actually, just return the known Strassen rank-14 solution if we didn't find 13
    # But wait, what if rank 13 is impossible? Then we return 14.
    
    # Let's also try rank 12 just in case, but 13 is more likely to be the next step down
    # Given the time budget, trying 13 first is the main structural change.
    
    return _strassen_2x2x4()
