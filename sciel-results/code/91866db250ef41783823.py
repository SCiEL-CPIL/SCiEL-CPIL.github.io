import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s to leave margin
    
    # Target tensor T[i,j,k] = 1
    # We want sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    
    # Schoolbook fallback
    rank_school = n * m * p
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
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Precompute index arrays for vectorized operations
    # For U update: for each (i,j), we need A[k,c] = V[j*p+k,c] * W[k*n+i,c]
    # For V update: for each (j,k), we need A[i,c] = U[i*m+j,c] * W[k*n+i,c]
    # For W update: for each (k,i), we need A[j,c] = U[i*m+j,c] * V[j*p+k,c]
    
    # Try to find lower rank
    min_rank_to_try = max(1, rank_school - 20)
    
    for r in range(rank_school - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        max_restarts = 10 if r > 100 else 30
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(restart * 100 + r)
            
            # Initialize factors with structured patterns
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS Loop
            converged = False
            for it in range(50):
                # Update U: for each (i,j), solve A u = ones(p)
                # A[k,c] = V[j*p+k,c] * W[k*n+i,c]
                new_U = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # Vectorized: A[k,c] = V[j*p+k,c] * W[k*n+i,c]
                        V_slice = V[j*p:(j+1)*p, :]  # (p, r)
                        W_slice = np.array([W[k*n+i, :] for k in range(p)])  # (p, r)
                        A = V_slice * W_slice  # (p, r)
                        
                        try:
                            u_sol, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            new_U[i * m + j, :] = u_sol
                        except:
                            pass
                U = new_U
                
                # Update V: for each (j,k), solve A v = ones(n)
                # A[i,c] = U[i*m+j,c] * W[k*n+i,c]
                new_V = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        U_slice = np.array([U[i*m+j, :] for i in range(n)])  # (n, r)
                        W_slice = np.array([W[k*n+i, :] for i in range(n)])  # (n, r)
                        A = U_slice * W_slice  # (n, r)
                        
                        try:
                            v_sol, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            new_V[j * p + k, :] = v_sol
                        except:
                            pass
                V = new_V
                
                # Update W: for each (k,i), solve A w = ones(m)
                # A[j,c] = U[i*m+j,c] * V[j*p+k,c]
                new_W = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        U_slice = np.array([U[i*m+j, :] for j in range(m)])  # (m, r)
                        V_slice = np.array([V[j*p+k, :] for j in range(m)])  # (m, r)
                        A = U_slice * V_slice  # (m, r)
                        
                        try:
                            w_sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            new_W[k * n + i, :] = w_sol
                        except:
                            pass
                W = new_W
                
                # Check residual
                residual = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            residual += (val - 1)**2
                
                if residual < 1e-10:
                    converged = True
                    break
                    
            if converged:
                # Try to snap to half-integers
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Verify exactness of snapped version
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_snapped[i * m + j, :] * V_snapped[j * p + k, :] * W_snapped[k * n + i, :])
                            if abs(val - 1) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    best_U = U_snapped
                    best_V = V_snapped
                    best_W = W_snapped
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
    
    # Convert to list of lists for output
    if best_rank == rank_school:
        U_out = [[int(x) for x in row] for row in U_school]
        V_out = [[int(x) for x in row] for row in V_school]
        W_out = [[int(x) for x in row] for row in W_school]
    else:
        U_out = [[float(x) for x in row] for row in best_U]
        V_out = [[float(x) for x in row] for row in best_V]
        W_out = [[float(x) for x in row] for row in best_W]
        
    return {"U": U_out, "V": V_out, "W": W_out}
