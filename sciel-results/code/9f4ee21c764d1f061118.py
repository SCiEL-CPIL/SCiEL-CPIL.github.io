"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build the tensor T[i,j,k] = 1 for all valid i,j,k
    # We need: for all i,j,k: 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100s out of 120s allowed
    
    # Schoolbook as fallback
    U_fb = np.zeros((n * m, schoolbook_rank))
    V_fb = np.zeros((m * p, schoolbook_rank))
    W_fb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*p, m*p, n*m) = max(18, 30, 15) = 30
    lower_bound = max(n * p, m * p, n * m)
    
    # Try ranks from lower_bound + some margin up to schoolbook_rank - 1
    # We'll try a range of ranks, starting from promising values
    
    # Precompute index mappings
    # For each (i,j,k), the equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try ALS for various ranks
    # We'll try ranks in a strategic order
    
    # Known: for (2,2,2) rank 7 works. For (3,3,3) rank 23 works.
    # For (3,5,6), let's try ranks from 30 to 89
    
    # To be efficient, try a subset of ranks first
    ranks_to_try = list(range(lower_bound, min(schoolbook_rank, 50))) + \
                   list(range(50, schoolbook_rank, 2))
    ranks_to_try = sorted(set(ranks_to_try))
    
    np.random.seed(42)
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try multiple initializations for this rank
        max_inits = 5 if r < 50 else 2
        if time.time() > deadline - 5:
            max_inits = 1
        
        for init in range(max_inits):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # ALS iterations
            max_iter = 50 if r < 50 else 30
            converged = False
            
            for iteration in range(max_iter):
                if time.time() > deadline - 3:
                    break
                
                # Update U: for each (i,j), solve linear system
                # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
                # This is: U[i*m+j, :] @ diag(V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
                # Actually: for fixed (i,j), we have p equations (one for each k):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Let M_k = V[j*p+k, :] * W[k*n+i, :] (element-wise product, shape (r,))
                # Then: U[i*m+j, :] @ M_k = 1 for all k
                # So: U[i*m+j, :] @ M^T = 1^T where M is (p, r)
                
                # Build M for this (i,j)
                M = np.zeros((p, r))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                
                # Solve: U[i*m+j, :] @ M^T = ones(p)
                # This is a least squares problem
                try:
                    sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U[i * m + j, :] = sol
                except:
                    pass
                
                # Similarly update V and W
                
                # Update V: for each (j,k), 
                # sum_c V[j*p+k, c] * (U[i*m+j, c] * W[k*n+i, c]) = 1 for all i
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, r))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                            V[j * p + k, :] = sol
                        except:
                            pass
                
                # Update W: for each (k,i),
                # sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1 for all j
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, r))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                            W[k * n + i, :] = sol
                        except:
                            pass
                
                # Check convergence
                if iteration % 10 == 0 or iteration == max_iter - 1:
                    # Compute error
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                                err = abs(val - 1)
                                if err > max_err:
                                    max_err = err
                    if max_err < 1e-10:
                        converged = True
                        break
            
            if converged or time.time() > deadline - 3:
                # Check if this is a valid solution
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            err = abs(val - 1)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-8:
                    # Try to round to half-integers
                    # Scale by 2 and round
                    U2 = np.round(U * 2) / 2
                    V2 = np.round(V * 2) / 2
                    W2 = np.round(W * 2) / 2
                    
                    # Check if rounded version is valid
                    max_err_rounded = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U2[i * m + j, :] * V2[j * p + k, :] * W2[k * n + i, :])
                                err = abs(val - 1)
                                if err > max_err_rounded:
                                    max_err_rounded = err
                    
                    if max_err_rounded < 1e-6:
                        # Valid half-integer solution!
                        if r < best_rank:
                            best_rank = r
                            best_result = {
                                "U": U2.tolist(),
                                "V": V2.tolist(),
                                "W": W2.tolist()
                            }
                            # Found a better solution, but keep searching for even better
                            break  # Break inner init loop, continue to next rank
            
            if time.time() > deadline - 5:
                break
        
        if time.time() > deadline - 5:
            break
    
    return best_result
# EVOLVE-BLOCK-END
