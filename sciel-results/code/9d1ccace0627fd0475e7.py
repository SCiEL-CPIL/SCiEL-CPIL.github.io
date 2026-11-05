import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook fallback
    rank_sb = n * m * p  # 96
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    def verify(U, V, W, tol=1e-8):
        """Verify the decomposition exactly."""
        r = U.shape[1]
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def alternating_minimize(U, V, W, max_iters=200, tol=1e-10, deadline=None):
        """Alternating least squares to refine the decomposition."""
        r = U.shape[1]
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M_mat = np.zeros((p, r))
                    for k in range(p):
                        M_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :], _, _, _ = np.linalg.lstsq(M_mat.T, np.ones(p), rcond=None)
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M_mat = np.zeros((n, r))
                    for i in range(n):
                        M_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :], _, _, _ = np.linalg.lstsq(M_mat.T, np.ones(n), rcond=None)
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M_mat = np.zeros((m, r))
                    for j in range(m):
                        M_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :], _, _, _ = np.linalg.lstsq(M_mat.T, np.ones(m), rcond=None)
                    except:
                        pass
            
            # Check convergence
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        error += (val - 1.0)**2
            if error < tol:
                break
        
        # Final error check
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    error += (val - 1.0)**2
        return error < 1e-8, U, V, W
    
    # Try ranks from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    lower_bound = max(n*m, m*p, p*n)
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 25:
            break
        if target_rank >= best_rank:
            continue
        
        found = False
        # Multiple random initializations
        for trial in range(8):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(12345 + trial * 7919 + target_rank * 31)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Scale to help convergence
            U *= np.sqrt(target_rank / (n*m))
            V *= np.sqrt(target_rank / (m*p))
            W *= np.sqrt(target_rank / (p*n))
            
            converged, U, V, W = alternating_minimize(U, V, W, max_iters=150, deadline=deadline)
            
            if converged and verify(U, V, W):
                best_rank = target_rank
                best_result = (U.copy(), V.copy(), W.copy())
                found = True
                break
        
        if found:
            # Try to go even lower
            continue
    
    U_list = best_result[0].tolist()
    V_list = best_result[1].tolist()
    W_list = best_result[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
