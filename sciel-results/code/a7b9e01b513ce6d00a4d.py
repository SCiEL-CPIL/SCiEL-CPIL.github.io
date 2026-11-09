import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def _compute_residual(U, V, W, n, m, p, r):
    """Compute the sum of squared residuals for the tensor decomposition."""
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                residual += (val - 1.0) ** 2
    return residual

def _check_exactness(U, V, W, n, m, p, r, tol=1e-9):
    """Check if the decomposition is exact within tolerance."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # 115s to leave margin for 120s limit
    
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
    
    # Try to find lower rank
    # We search from rank_school - 1 down to some limit
    min_rank_to_try = max(1, rank_school - 30)  # Try down to 75
    
    for r in range(rank_school - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 15:
            break
            
        # Try random restarts for this rank
        found = False
        max_restarts = 15 if r > 100 else 30
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(restart * 100 + r)
            
            # Initialize factors
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS Loop with improved convergence check
            converged = False
            prev_residual = float('inf')
            
            for it in range(150):
                # Update U
                new_U = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        
                        try:
                            u_sol, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            new_U[i * m + j, :] = u_sol
                        except:
                            pass
                U = new_U
                
                # Update V
                new_V = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        
                        try:
                            v_sol, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            new_V[j * p + k, :] = v_sol
                        except:
                            pass
                V = new_V
                
                # Update W
                new_W = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        
                        try:
                            w_sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            new_W[k * n + i, :] = w_sol
                        except:
                            pass
                W = new_W
                
                # Check convergence
                residual = _compute_residual(U, V, W, n, m, p, r)
                
                if residual < 1e-12:
                    converged = True
                    break
                
                if abs(prev_residual - residual) < 1e-14 and it > 10:
                    # Stagnation check
                    if residual < 1e-8:
                        converged = True
                    break
                prev_residual = residual
                
            if converged:
                # Try to snap to half-integers
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Verify exactness of snapped version
                if _check_exactness(U_snapped, V_snapped, W_snapped, n, m, p, r):
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
