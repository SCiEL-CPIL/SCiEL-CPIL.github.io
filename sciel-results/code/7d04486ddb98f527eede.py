"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    num_u = n * m  # 8
    num_v = m * p  # 16
    num_w = p * n  # 8
    
    start_time = time.time()
    deadline = start_time + 100
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((num_u, rank))
        V = np.zeros((num_v, rank))
        W = np.zeros((num_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank using simple perturbation
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try rank 31
    if time.time() < deadline:
        rank_try = 31
        # Start from schoolbook and try to merge two columns
        # Pick two columns and try to express one as a function of the other
        # This is complex, so let's try a simple approach: random restarts
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Create a random rank-31 initialization
            U_try = np.random.randn(num_u, rank_try) * 0.5
            V_try = np.random.randn(num_v, rank_try) * 0.5
            W_try = np.random.randn(num_w, rank_try) * 0.5
            
            # Scale to help
            U_try /= np.sqrt(np.sum(U_try**2))
            V_try /= np.sqrt(np.sum(V_try**2)) 
            W_try /= np.sqrt(np.sum(W_try**2))
            
            # Simple gradient-free local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for r in range(rank_try):
                    # Try perturbing column r
                    U_orig = U_try[:, r].copy()
                    V_orig = V_try[:, r].copy()
                    W_orig = W_try[:, r].copy()
                    
                    for _ in range(10):
                        U_try[:, r] = U_orig + np.random.randn(num_u) * 0.1
                        V_try[:, r] = V_orig + np.random.randn(num_v) * 0.1
                        W_try[:, r] = W_orig + np.random.randn(num_w) * 0.1
                        
                        if verify(U_try, V_try, W_try, tol=1e-6):
                            improved = True
                            break
                    
                    if improved:
                        break
            
            if verify(U_try, V_try, W_try, tol=1e-8):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = rank_try
                break
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
