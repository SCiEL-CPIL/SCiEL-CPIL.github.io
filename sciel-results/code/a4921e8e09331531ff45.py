"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 24, 16
    schoolbook_rank = n * m * p  # 48
    
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    def check_decomposition(U, V, W, r, tol=1e-10):
        """Check if U, V, W of rank r exactly decompose the tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def alternating_minimize(U, V, W, r, max_iter=2000, tol=1e-12):
        """Alternating minimization to find rank-r decomposition."""
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Fix V, W, optimize U
            new_U = U.copy()
            for i in range(n):
                for j in range(m):
                    # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    M = V[j*p:(j+1)*p, :] * W[np.arange(p)*n+i, :].T
                    # Actually need: M[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        new_U[i*m+j, :] = u
                    except:
                        pass
            
            # Fix U, W, optimize V
            new_V = V.copy()
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        new_V[j*p+k, :] = v
                    except:
                        pass
            
            # Fix U, V, optimize W
            new_W = W.copy()
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        new_W[k*n+i, :] = w
                    except:
                        pass
            
            U, V, W = new_U, new_V, new_W
            
            # Check convergence
            max_change = max(np.max(np.abs(U - new_U)), 
                           np.max(np.abs(V - new_V)), 
                           np.max(np.abs(W - new_W)))
            if max_change < tol:
                break
        
        return U, V, W
    
    def try_rank_with_als(r, num_trials=30):
        """Try to find rank-r decomposition using alternating minimization."""
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(trial * 7919 + r * 104729)
            scale = 1.0 / np.sqrt(r)
            U_init = np.random.randn(nm, r) * scale
            V_init = np.random.randn(mp, r) * scale
            W_init = np.random.randn(pn, r) * scale
            
            U, V, W = alternating_minimize(U_init, V_init, W_init, r, max_iter=1000)
            
            if check_decomposition(U, V, W, r):
                return U, V, W
        
        return None
    
    # Fallback: Schoolbook
    rank = schoolbook_rank
    U_sb = np.zeros((nm, rank))
    V_sb = np.zeros((mp, rank))
    W_sb = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank
    
    # Try ranks from lower bound up
    lower_bound = max(nm, mp, pn)  # 24
    
    for r in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 20:
            break
        
        result = try_rank_with_als(r, num_trials=50)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
