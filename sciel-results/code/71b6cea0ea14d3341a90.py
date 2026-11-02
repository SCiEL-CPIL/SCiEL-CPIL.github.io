import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    # Target tensor T[i,j,k] = 1 for all valid indices
    T = np.ones((n, m, p))
    
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill

    def compute_error(U, V, W, target_rank):
        """Compute the error of the tensor decomposition."""
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        
        # result[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.linalg.norm(result - T)

    def verify_and_round(U, V, W, target_rank):
        """Round to half-integers and verify exactness."""
        # Try rounding to nearest half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check exactness with high precision
        err = compute_error(U_r, V_r, W_r, target_rank)
        if err < 1e-9:
            return U_r.reshape(-1, target_rank), V_r.reshape(-1, target_rank), W_r.reshape(-1, target_rank)
        
        # If half-integer rounding fails, try integer rounding (less likely but possible)
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        err = compute_error(U_r, V_r, W_r, target_rank)
        if err < 1e-9:
            return U_r.reshape(-1, target_rank), V_r.reshape(-1, target_rank), W_r.reshape(-1, target_rank)
            
        return None

    def polish_solution(U, V, W, target_rank, max_iters=200):
        """Polish the solution using Least Squares with Levenberg-Marquardt."""
        
        def objective(params):
            U_flat = params[:n*m*target_rank].reshape(n, m, target_rank)
            V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m, p, target_rank)
            W_flat = params[n*m*target_rank + m*p*target_rank:].reshape(p, n, target_rank)
            
            result = np.einsum('ijr,jkr,kir->ijk', U_flat, V_flat, W_flat)
            diff = result - T
            return diff.flatten()
        
        initial_params = np.concatenate([
            U.flatten(), 
            V.flatten(), 
            W.flatten()
        ])
        
        try:
            res = opt.least_squares(
                objective, 
                initial_params, 
                method='lm', # Levenberg-Marquardt
                max_nfev=max_iters
            )
            
            # Check if we converged to a very low error
            if res.cost < 1e-15:
                U_new = res.x[:n*m*target_rank].reshape(n, m, target_rank)
                V_new = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m, p, target_rank)
                W_new = res.x[n*m*target_rank + m*p*target_rank:].reshape(p, n, target_rank)
                return U_new.reshape(-1, target_rank), V_new.reshape(-1, target_rank), W_new.reshape(-1, target_rank)
        except:
            pass
        return U, V, W

    def als_decompose(target_rank, max_iters=20, tol=1e-6, seed=None):
        """Alternating Least Squares for tensor decomposition to get close to a solution."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
            
        # Initialize with small random values
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
        
        prev_err = float('inf')
        
        for iter_ in range(max_iters):
            if time.time() - start_time > deadline:
                return None, None, None, None
                
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A.T, T[i, j, :].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, target_rank))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(B.T, T[:, j, k].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, target_rank))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(C.T, T[i, :, k].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            # Check convergence
            err = compute_error(U, V, W, target_rank)
            if err < tol:
                return U, V, W, err
            if abs(err - prev_err) < 1e-8:
                break
            prev_err = err
            
        err = compute_error(U, V, W, target_rank)
        return U, V, W, err

    # Try ranks from 14 upward (known minimal rank for 2x3x3 is 14)
    # We prioritize lower ranks as they are better.
    for target_rank in range(14, R_school + 1):
        if time.time() - start_time > deadline:
            break
            
        found = False
        # Use more restarts for lower ranks as they are harder
        max_restarts = 200 if target_rank <= 14 else 50
        
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break
                
            U, V, W, err = als_decompose(target_rank, max_iters=10, tol=1e-4, seed=restart)
            
            if U is None:
                break
            
            # If ALS got close, try to polish with LM
            if err < 1.0:
                U, V, W = polish_solution(U, V, W, target_rank)
                err = compute_error(U, V, W, target_rank)
                
                if err < 1e-8:
                    # Attempt to round and verify
                    rounded = verify_and_round(U, V, W, target_rank)
                    if rounded is not None:
                        best_U, best_V, best_W = rounded
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            # If we found a solution for this rank, we can stop since we iterate from low to high
            # But we should check if we can go lower? No, we start from 14.
            # So if we found rank 14, we are done.
            break
        else:
            continue
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
