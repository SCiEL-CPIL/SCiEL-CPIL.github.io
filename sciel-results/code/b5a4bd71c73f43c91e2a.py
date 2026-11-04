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
    deadline = start_time + 140.0  # 10s safety margin
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_seed
    
    def compute_error(U, V, W, R):
        """Compute the max absolute error of the decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :R], V[j*p+k, :R] * W[k*n+i, :R])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def als_step(U, V, W, R, max_iters=30, tol=1e-12):
        """Alternating least squares to minimize error."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), solve B @ U[i*m+j] = ones(p)
            # where B[k, r] = V[j*p+k, r] * W[k*n+i, r]
            for i in range(n):
                for j in range(m):
                    B = np.zeros((p, R))
                    for k in range(p):
                        B[k, :] = V[j*p+k, :R] * W[k*n+i, :R]
                    try:
                        U[i*m+j, :R] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve B @ V[j*p+k] = ones(n)
            # where B[i, r] = U[i*m+j, r] * W[k*n+i, r]
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, R))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :R] * W[k*n+i, :R]
                    try:
                        V[j*p+k, :R] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve B @ W[k*n+i] = ones(m)
            # where B[j, r] = U[i*m+j, r] * V[j*p+k, r]
            for k in range(p):
                for i in range(n):
                    B = np.zeros((m, R))
                    for j in range(m):
                        B[j, :] = U[i*m+j, :R] * V[j*p+k, :R]
                    try:
                        W[k*n+i, :R] = np.linalg.lstsq(B, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            if it % 3 == 2:
                err = compute_error(U, V, W, R)
                if err < tol:
                    break
                if abs(err - prev_err) < 1e-14:
                    break
                prev_err = err
        
        return U, V, W
    
    def try_round(U, V, W, R):
        """Try rounding to half-integers and verify."""
        U_r = np.round(2 * U) / 2
        V_r = np.round(2 * V) / 2
        W_r = np.round(2 * W) / 2
        
        err = compute_error(U_r, V_r, W_r, R)
        if err < 1e-9:
            return U_r, V_r, W_r, True
        return None, None, None, False
    
    # Try ranks from rank_seed - 1 down to 80
    for R in range(rank_seed - 1, 79, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        num_restarts = 8 if R > 90 else 12
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            seed_val = 100000 + R * 1000 + restart
            np.random.seed(seed_val)
            
            # Initialize close to schoolbook with perturbation
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            # Copy as much of schoolbook as possible
            n_copy = min(R, rank_seed)
            U[:, :n_copy] = U0[:, :n_copy]
            V[:, :n_copy] = V0[:, :n_copy]
            W[:, :n_copy] = W0[:, :n_copy]
            
            # Add perturbation to break degeneracy
            scale = 0.1 if R == rank_seed - 1 else 0.05
            U += scale * np.random.randn(n*m, R)
            V += scale * np.random.randn(m*p, R)
            W += scale * np.random.randn(p*n, R)
            
            # Run ALS
            U, V, W = als_step(U, V, W, R, max_iters=40)
            
            err = compute_error(U, V, W, R)
            
            if err < 1e-10:
                # Try rounding
                U_r, V_r, W_r, success = try_round(U, V, W, R)
                if success:
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = R
                    found = True
                    break
            
            # Also try without rounding if error is very small (for verification)
            if err < 1e-15:
                # Check if values are already close to half-integers
                U_diff = np.abs(U - np.round(2*U)/2)
                V_diff = np.abs(V - np.round(2*V)/2)
                W_diff = np.abs(W - np.round(2*W)/2)
                if np.max(U_diff) < 1e-6 and np.max(V_diff) < 1e-6 and np.max(W_diff) < 1e-6:
                    U_r, V_r, W_r = np.round(2*U)/2, np.round(2*V)/2, np.round(2*W)/2
                    err_r = compute_error(U_r, V_r, W_r, R)
                    if err_r < 1e-9:
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = R
                        found = True
                        break
        
        if found:
            break
    
    # Final verification
    final_err = compute_error(best_U, best_V, best_W, best_rank)
    if final_err > 1e-9:
        # Fall back to schoolbook
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank_seed
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
