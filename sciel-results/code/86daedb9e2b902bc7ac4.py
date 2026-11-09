import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    U_fb, V_fb, W_fb, R_fb = schoolbook()
    
    np.random.seed(42)
    
    def compute_error(U, V, W, R):
        """Compute sum of squared errors efficiently"""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        return err
    
    def try_snap(U, V, W, R, tol=1e-8):
        """Try to snap to half-integers and verify exactness"""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        snap_err = compute_error(U_snap, V_snap, W_snap, R)
        
        if snap_err < tol:
            return U_snap.copy(), V_snap.copy(), W_snap.copy()
        return None

    best_rank = R_fb
    best_factors = (U_fb, V_fb, W_fb)
    
    # Try ranks from 62 down to 54
    # Prioritize higher ranks (closer to schoolbook) as they are easier to find
    ranks_to_try = [62, 61, 60, 59, 58, 57, 56, 55, 54]
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        
        best_err_for_R = float('inf')
        best_U_V_W_for_R = None
        
        # More restarts for higher ranks
        n_restarts = 30 if R >= 60 else 20
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Initialize
            if restart % 3 == 0:
                U = np.random.randn(n * m, R)
                V = np.random.randn(m * p, R)
                W = np.random.randn(p * n, R)
            else:
                # Sparse initialization might help with structured tensors
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                for r in range(R):
                    idx_u = np.random.randint(0, n*m, 2)
                    idx_v = np.random.randint(0, m*p, 2)
                    idx_w = np.random.randint(0, p*n, 2)
                    U[idx_u, r] = np.random.choice([-1, 1])
                    V[idx_v, r] = np.random.choice([-1, 1])
                    W[idx_w, r] = np.random.choice([-1, 1])
            
            # ALS iterations
            max_iters = 80
            prev_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            MMt = M @ M.T
                            MMt += 1e-10 * np.eye(p)
                            x = np.linalg.solve(MMt, np.ones(p))
                            U[i * m + j, :] = M.T @ x
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, R))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            MMt = M @ M.T
                            MMt += 1e-10 * np.eye(n)
                            x = np.linalg.solve(MMt, np.ones(n))
                            V[j * p + k, :] = M.T @ x
                        except:
                            pass
                        
                # Update W
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, R))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            MMt = M @ M.T
                            MMt += 1e-10 * np.eye(m)
                            x = np.linalg.solve(MMt, np.ones(m))
                            W[k * n + i, :] = M.T @ x
                        except:
                            pass
                
                # Calculate error
                err = compute_error(U, V, W, R)
                
                if err < 1e-10:
                    break
                
                if err > prev_err * 1.05 and it > 5:
                    break
                prev_err = err
            
            # Check for snap-able solution
            result = try_snap(U, V, W, R)
            if result is not None:
                U_snap, V_snap, W_snap = result
                snap_err = compute_error(U_snap, V_snap, W_snap, R)
                if snap_err < best_err_for_R:
                    best_err_for_R = snap_err
                    best_U_V_W_for_R = (U_snap, V_snap, W_snap)
                
                # If we found a valid solution for this R, update global best and break
                # Since we are iterating R from high to low, the first valid one is the best for this pass
                if best_err_for_R < 1e-8:
                    best_rank = R
                    best_factors = best_U_V_W_for_R
                    break # Break restart loop, move to next R (lower)

    # Return the best found
    U, V, W = best_factors
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
