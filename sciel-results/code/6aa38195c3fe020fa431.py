import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1
    # Decomposition: sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1
    
    # Return schoolbook if search fails
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# Search for rank 10
def try_rank_10():
    n, m, p = DIMENSIONS
    rank = 10
    # Variables: U (4x10), V (6x10), W (6x10)
    # Total 160 vars
    
    # Use scipy.optimize if available, else simple random search
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None

    def objective(params):
        U = params[0:40].reshape(4, 10)
        V = params[40:100].reshape(6, 10)
        W = params[100:160].reshape(6, 10)
        
        # Compute residual
        # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We want this to be 1 for all i,j,k
        residual = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    residual += (s - 1)**2
        return residual

    # Start from a random initialization or schoolbook-like
    # Schoolbook has rank 12. We can take first 10 columns?
    # Or random.
    
    best_sol = None
    best_val = float('inf')
    
    start_time = time.time()
    timeout = 60  # seconds
    
    for restart in range(20):
        # Random initialization
        x0 = np.random.randn(160) * 0.1
        # Try to bias towards integers?
        
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < best_val:
            best_val = res.fun
            best_sol = res.x
            
        if time.time() - start_time > timeout:
            break
            
    if best_sol is None:
        return None
        
    if best_val < 1e-4:
        # Snap to half-integers
        U = best_sol[0:40].reshape(4, 10)
        V = best_sol[40:100].reshape(6, 10)
        W = best_sol[100:160].reshape(6, 10)
        
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_snap[i*m+j, r] * V_snap[j*p+k, r] * W_snap[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
    return None

def construct():
    # Try to find rank 10
    sol = try_rank_10()
    if sol:
        return sol
    # Fallback to schoolbook
    n, m, p = DIMENSIONS
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
