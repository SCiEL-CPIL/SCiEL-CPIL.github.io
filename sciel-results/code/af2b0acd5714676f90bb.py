import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_seed = n * m * p  # 96
    
    # Schoolbook solution
    U0 = np.zeros((n*m, rank_seed))
    V0 = np.zeros((m*p, rank_seed))
    W0 = np.zeros((p*n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    start_time = time.time()
    timeout = 140.0  # Leave 10s safety margin
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_seed
    
    def compute_error(U, V, W, R):
        """Compute the squared error of the decomposition."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :R], V[j*p+k, :R] * W[k*n+i, :R])
                    err += (val - 1.0)**2
        return err
    
    def als_step(U, V, W, R, max_iters=50):
        """Alternating least squares to minimize error."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > timeout - 5:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    B = np.zeros((p, R))
                    for k in range(p):
                        B[k, :] = V[j*p+k, :R] * W[k*n+i, :R]
                    try:
                        U[i*m+j, :R] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, R))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :R] * W[k*n+i, :R]
                    try:
                        V[j*p+k, :R] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    B = np.zeros((m, R))
                    for j in range(m):
                        B[j, :] = U[i*m+j, :R] * V[j*p+k, :R]
                    try:
                        W[k*n+i, :R] = np.linalg.lstsq(B, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Early stopping
            if it > 0 and it % 5 == 0:
                err = compute_error(U, V, W, R)
                if err < 1e-12:
                    break
        
        return U, V, W
    
    # Try ranks from rank_seed - 1 down to 85
    for R in range(rank_seed - 1, 84, -1):
        if time.time() - start_time > timeout - 10:
            break
        
        found = False
        for init_seed in range(5):
            if time.time() - start_time > timeout - 5:
                break
            
            np.random.seed(12345 + R * 100 + init_seed)
            
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            n_copy = min(R, rank_seed)
            U[:, :n_copy] = U0[:, :n_copy]
            V[:, :n_copy] = V0[:, :n_copy]
            W[:, :n_copy] = W0[:, :n_copy]
            
            # Perturb to break degeneracy
            U += 0.05 * np.random.randn(n*m, R)
            V += 0.05 * np.random.randn(m*p, R)
            W += 0.05 * np.random.randn(p*n, R)
            
            # Run ALS
            U, V, W = als_step(U, V, W, R, max_iters=40)
            
            err = compute_error(U, V, W, R)
            if err < 1e-10:
                # Try rounding to half-integers
                U_rounded = np.round(2 * U) / 2
                V_rounded = np.round(2 * V) / 2
                W_rounded = np.round(2 * W) / 2
                
                err_rounded = compute_error(U_rounded, V_rounded, W_rounded, R)
                if err_rounded < 1e-10:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = R
                    found = True
                    break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
