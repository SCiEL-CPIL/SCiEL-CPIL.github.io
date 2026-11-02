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
    deadline = start_time + 135  # Safety margin before 150s kill
    
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
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check exactness
        err = compute_error(U_r, V_r, W_r, target_rank)
        if err < 1e-9:
            return U_r.reshape(-1, target_rank), V_r.reshape(-1, target_rank), W_r.reshape(-1, target_rank)
        return None
    
    def polish_solution(U, V, W, target_rank, max_iters=200):
        """Polish the solution using Least Squares with tighter tolerance."""
        
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
                method='lm',
                max_nfev=max_iters,
                ftol=1e-15,
                xtol=1e-15,
                gtol=1e-15
            )
            
            if res.cost < 1e-15:
                U_new = res.x[:n*m*target_rank].reshape(n, m, target_rank)
                V_new = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m, p, target_rank)
                W_new = res.x[n*m*target_rank + m*p*target_rank:].reshape(p, n, target_rank)
                return U_new.reshape(-1, target_rank), V_new.reshape(-1, target_rank), W_new.reshape(-1, target_rank)
        except:
            pass
        return U, V, W
    
    def als_decompose(target_rank, max_iters=100, tol=1e-10, seed=None, polish=True):
        """Alternating Least Squares for tensor decomposition."""
        rng = np.random.RandomState(seed)
            
        # Initialize with small random values
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
        
        prev_err = float('inf')
        
        for iter_ in range(max_iters):
            if time.time() > deadline:
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
                if polish:
                    U, V, W = polish_solution(U, V, W, target_rank)
                    err = compute_error(U, V, W, target_rank)
                return U, V, W, err
            if abs(err - prev_err) < 1e-12:
                if err < 1e-6:
                    if polish:
                        U, V, W = polish_solution(U, V, W, target_rank)
                        err = compute_error(U, V, W, target_rank)
                    return U, V, W, err
                break
            prev_err = err
            
        # Final polish if we are close
        err = compute_error(U, V, W, target_rank)
        if err < 1e-3 and polish:
            U, V, W = polish_solution(U, V, W, target_rank)
            err = compute_error(U, V, W, target_rank)
            
        return U, V, W, err
    
    def try_rank_with_structured_init(target_rank, max_restarts=50):
        """Try to find decomposition with various initialization strategies."""
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None, None, None, None
            
            if restart % 5 == 0:
                # Random Gaussian
                seed = restart
            elif restart % 5 == 1:
                # Sparse binary-like init
                seed = restart + 1000
            elif restart % 5 == 2:
                # Small values
                seed = restart + 2000
            elif restart % 5 == 3:
                # Larger values
                seed = restart + 3000
            else:
                # Mixed
                seed = restart + 4000
            
            U, V, W, err = als_decompose(target_rank, max_iters=50, tol=1e-8, seed=seed, polish=True)
            
            if U is None:
                break
                
            if err < 1e-8:
                # Attempt to round and verify
                rounded = verify_and_round(U, V, W, target_rank)
                if rounded is not None:
                    return rounded[0], rounded[1], rounded[2], 0.0
        
        return None, None, None, None
    
    # Try ranks from 14 upward (known minimal rank for 2x3x3 is 14)
    for target_rank in range(14, R_school + 1):
        if time.time() > deadline:
            break
            
        found = False
        max_restarts = 30 if target_rank <= 14 else 15
        
        U_new, V_new, W_new, err = try_rank_with_structured_init(target_rank, max_restarts)
        
        if U_new is not None:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = target_rank
            found = True
        
        if found:
            continue
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
