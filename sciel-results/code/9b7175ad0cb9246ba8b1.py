import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Schoolbook solution
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    def verify(U, V, W, tol=1e-8):
        """Check exact decomposition."""
        n, m, p = DIMENSIONS
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def tensor_error(U, V, W):
        """Compute sum of squared errors."""
        n, m, p = DIMENSIONS
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        return err
    
    def als_step(U, V, W, n, m, p, r, lr=1.0):
        """One ALS iteration: update U, V, W in sequence."""
        U_new = np.zeros_like(U)
        V_new = np.zeros_like(V)
        W_new = np.zeros_like(W)
        
        # Update U: for each row i*m+j, solve least squares
        for i in range(n):
            for j in range(m):
                # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # Fix V and W, solve for U[i*m+j, :]
                # This is: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
                # Build matrix A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A @ U_row = ones(p)
                try:
                    U_row, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U_new[i*m+j, :] = U_row
                except:
                    U_new[i*m+j, :] = U[i*m+j, :]
        
        # Update V: for each row j*p+k
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U_new[i*m+j, :] * W[k*n+i, :]
                try:
                    V_row, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V_new[j*p+k, :] = V_row
                except:
                    V_new[j*p+k, :] = V[j*p+k, :]
        
        # Update W: for each row k*n+i
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U_new[i*m+j, :] * V_new[j*p+k, :]
                try:
                    W_row, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W_new[k*n+i, :] = W_row
                except:
                    W_new[k*n+i, :] = W[k*n+i, :]
        
        return U_new, V_new, W_new
    
    # Strategy: Start from schoolbook, try to reduce rank by finding
    # linear dependencies and using ALS to refine
    
    # Try ranks from 83 down
    for target_rank in range(rank_school - 1, max(70, rank_school - 15), -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(8):
            if time.time() > deadline - 8:
                    break
            
            np.random.seed(42 + trial * 1000 + target_rank)
            
            # Initialize: take first target_rank columns of schoolbook, 
            # then add random perturbation
            U_init = U_school[:, :target_rank].copy()
            V_init = V_school[:, :target_rank].copy()
            W_init = W_school[:, :target_rank].copy()
            
            # Add small random perturbation to break symmetry
            U_init += np.random.randn(n*m, target_rank) * 0.01
            V_init += np.random.randn(m*p, target_rank) * 0.01
            W_init += np.random.randn(p*n, target_rank) * 0.01
            
            # Run ALS iterations
            U_cur, V_cur, W_cur = U_init, V_init, W_init
            prev_err = tensor_error(U_cur, V_cur, W_cur)
            
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                U_cur, V_cur, W_cur = als_step(U_cur, V_cur, W_cur, n, m, p, target_rank)
                curr_err = tensor_error(U_cur, V_cur, W_cur)
                if abs(prev_err - curr_err) < 1e-14:
                    break
                prev_err = curr_err
            
            # Check if we found a valid solution
            if verify(U_cur, V_cur, W_cur, tol=1e-6):
                best_rank = target_rank
                best_U, best_V, best_W = U_cur, V_cur, W_cur
                found = True
                break
        
        if found:
            break
    
    # If we found improvement, try to polish with tighter tolerance
    if best_rank < rank_school:
        # Polish pass
        U_pol, V_pol, W_pol = best_U.copy(), best_V.copy(), best_W.copy()
        for it in range(100):
            if time.time() > deadline - 2:
                break
            U_pol, V_pol, W_pol = als_step(U_pol, V_pol, W_pol, n, m, p, best_rank)
        
        if verify(U_pol, V_pol, W_pol, tol=1e-8):
            best_U, best_V, best_W = U_pol, V_pol, W_pol
        else:
            # Keep original if polish didn't help
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
