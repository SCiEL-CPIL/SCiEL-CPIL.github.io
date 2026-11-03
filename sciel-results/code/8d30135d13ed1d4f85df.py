# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin

    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Helper to check validity and snap to integers
    def check_and_snap(U, V, W, tol=1e-5):
        # Snap to nearest integer or half-integer if close
        # The problem allows integer/half-integer entries.
        # Often solutions are integers.
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Verify exactness
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(U.shape[1]):
                        total += U_snap[i*m+j, c] * V_snap[j*p+k, c] * W_snap[k*n+i, c]
                    if abs(total - 1.0) > tol:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {
                "U": U_snap.tolist(), 
                "V": V_snap.tolist(), 
                "W": W_snap.tolist()
            }
        return None

    # Helper to compute error
    def error_func(params, rank):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        err = 0.0
        # Sample a subset of entries if rank is large to speed up? 
        # No, n=2, m=3, p=9 is small. 54 equations.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (total - 1.0)**2
        return err

    best_result = None
    best_rank = schoolbook_rank

    # Try ranks from a lower bound up to schoolbook_rank - 1
    # Lower bound is at least max(n*m, m*p, p*n) ? No, rank >= max dims usually, but for matmul it's higher.
    # Let's try from 40 to 53.
    for target_rank in range(40, schoolbook_rank):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
                
            # Initialize with small random values
            np.random.seed(attempt * 1000 + target_rank)
            init_params = np.random.uniform(-1, 1, size=(n*m + m*p + p*n) * target_rank)
            
            # Use Nelder-Mead for local optimization
            try:
                res = minimize(error_func, init_params, args=(target_rank,), method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                
                if res.fun < 1e-6:
                    # Extract U, V, W
                    U = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    result = check_and_snap(U, V, W)
                    if result:
                        best_rank = target_rank
                        best_result = result
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            # Since we iterate rank increasing, the first one found is the lowest rank in this range
            # But we should check if we can go lower? 
            # The loop goes 40, 41, ... so 40 is the lowest we try first.
            # If we find 40, we break.
            break

    if best_result is None:
        # Fall back to schoolbook
        rank = schoolbook_rank
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
    
    return best_result
# EVOLVE-BLOCK-END
