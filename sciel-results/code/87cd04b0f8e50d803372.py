import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    # Build the target tensor T[i, j, k] = 1 for all (i,j,k)
    # T has shape (n, m, p)
    T = np.ones((n, m, p))
    
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
    
    start_time = time.time()
    deadline = 130  # Safety margin before 150s kill
    
    # We know R(2,3,3) = 14 from literature
    # Try ranks from 14 upward
    for target_rank in range(14, R_school + 1):
        if time.time() - start_time > deadline:
            break
        
        found = False
        max_restarts = 30 if target_rank <= 15 else 15
        
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break
            
            np.random.seed(restart * 1000 + target_rank)
            
            # Initialize randomly
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Normalize columns to avoid scale issues
            for r in range(target_rank):
                norms = [np.linalg.norm(U[:, r]), np.linalg.norm(V[:, r]), np.linalg.norm(W[:, r])]
                if any(norm < 1e-10 for norm in norms):
                    continue
                total_norm = np.prod(norms) ** (1/3)
                if total_norm > 1e-10:
                    U[:, r] /= (norms[0] / total_norm)
                    V[:, r] /= (norms[1] / total_norm)
                    W[:, r] /= (norms[2] / total_norm)
            
            # ALS iterations
            converged = False
            for iteration in range(200):
                if time.time() - start_time > deadline:
                    break
                
                # Update U: for each (i,j), solve least squares
                # T[i,j,:] = sum_r U[i*m+j, r] * (V[j,:] * W[:,i])
                # For fixed (i,j): T[i,j,:] = U[i*m+j,:] @ diag(V[j,:] * W[:,i])
                # Actually: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                
                for i in range(n):
                    for j in range(m):
                        # Build the linear system for U[i*m+j, :]
                        # T[i,j,k] = sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])
                        # Let A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # Then T[i,j,:] = U[i*m+j,:] @ A.T
                        # So U[i*m+j,:] = T[i,j,:] @ pinv(A.T)
                        
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        
                        # Solve U_row @ A.T = T[i,j,:]
                        # U_row = T[i,j,:] @ pinv(A.T)
                        try:
                            A_T = A.T  # shape (target_rank, p)
                            # U_row (1, target_rank) @ A_T (target_rank, p) = T[i,j,:] (p,)
                            # U_row = T[i,j,:] @ pinv(A_T)
                            U[i * m + j, :] = np.linalg.lstsq(A_T, T[i, j, :].reshape(-1, 1), rcond=None)[0].flatten()
                        except:
                            pass
                
                # Update V: for each (j,k), solve least squares
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed (j,k): T[:,j,k] = sum_r V[j*p+k,r] * (U[:,j,:] * W[k*n,:])
                # Let B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # Then T[:,j,k] = V[j*p+k,:] @ B.T
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, target_rank))
                        for i in range(n):
                            B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        
                        B_T = B.T  # shape (target_rank, n)
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(B_T, T[:, j, k].reshape(-1, 1), rcond=None)[0].flatten()
                        except:
                            pass
                
                # Update W: for each (k,i), solve least squares
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed (k,i): T[i,:,k] = sum_r W[k*n+i,r] * (U[i*m,:]*V[:,k,:])
                # Let C[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # Then T[i,:,k] = W[k*n+i,:] @ C.T
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((m, target_rank))
                        for j in range(m):
                            C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        
                        C_T = C.T  # shape (target_rank, m)
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(C_T, T[i, :, k].reshape(-1, 1), rcond=None)[0].flatten()
                        except:
                            pass
                
                # Check convergence
                # Compute error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            error += (val - T[i, j, k]) ** 2
                error = np.sqrt(error)
                
                if error < 1e-6:
                    converged = True
                    break
                
                # Early stopping if error is not decreasing
                if iteration > 50 and error > 1.0:
                    break
            
            if converged or error < 1e-5:
                # Verify the solution
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            max_err = max(max_err, abs(val - T[i, j, k]))
                
                if max_err < 1e-5:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    # Round to nearest half-integer if possible
    def round_to_half(arr):
        rounded = np.round(arr * 2) / 2
        # Check if rounding preserves accuracy
        max_diff = np.max(np.abs(rounded - arr))
        if max_diff < 1e-4:
            return rounded
        return arr
    
    best_U = round_to_half(best_U)
    best_V = round_to_half(best_V)
    best_W = round_to_half(best_W)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
