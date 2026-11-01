import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s margin
    
    # Precompute indices for the tensor
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def residual(params):
        # params: flattened U, V, W
        # U: (n*m, R)
        # V: (m*p, R)
        # W: (p*n, R)
        R = len(params) // (n*m + m*p + p*n)
        if R == 0: return np.array([0])
        
        U = params[0:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        
        # Reshape to 3D for einsum
        # U: (n, m, R)
        # V: (m, p, R)
        # W: (p, n, R)
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # S[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
        # einsum: 'ijr,jkr,kir->ijk'
        S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        return S.flatten() - 1.0

    best_solution = None
    
    # Try ranks from 45 up to 53 to find the MINIMAL rank
    for R in range(45, 54):
        if time.time() > deadline: break
        
        # Number of variables
        num_vars = R * (n*m + m*p + p*n)
        
        # We will try a few random restarts
        # Increase restarts if we have time? 
        # Let's try 30 restarts for lower ranks as they are harder.
        n_restarts = 30
        for restart in range(n_restarts):
            if time.time() > deadline: break
            
            np.random.seed(restart * 100 + R)
            
            # Initialize
            # Use smaller initialization to help convergence?
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            # Gauge fixing: 
            # Fix the first column of U (rank index 0) to 1s?
            # Or fix the norm of the first row?
            # Let's try fixing U[0, r] = 1 for all r (first row, all rank cols)
            # This assumes the first row of U is non-zero in all rank components.
            U[0, :] = 1.0
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                sol = least_squares(residual, params, method='lm', max_nfev=2000)
            except:
                continue
                
            if sol.success or np.linalg.norm(sol.fun) < 1e-8:
                # Reconstruct U, V, W
                U_sol = sol.x[0:n*m*R].reshape(n*m, R)
                V_sol = sol.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_sol = sol.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Try to snap to nearest 0.5
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                # Verify exact decomposition with snapped values
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i*m + j
                            v_idx = j*p + k
                            w_idx = k*n + i
                            s = np.dot(U_snap[u_idx, :] * V_snap[v_idx, :], W_snap[w_idx, :])
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    best_solution = (U_snap, V_snap, W_snap, R)
                    # Found a solution for this R. Since we iterate up, this is the minimal found so far.
                    # We can return immediately? 
                    # The task asks for rank-minimal. If we find 45, it's likely the min (or close).
                    # If we find 48, we should check if 47 is possible? 
                    # But we are iterating 45, 46, 47... 
                    # So if we find 45, we return. If we find 46, we return 46 only if 45 failed.
                    # So we can return immediately upon finding a valid solution for the current R.
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
            # If we found a valid solution, we would have returned.
            # So we continue to next restart.


    if best_solution:
        U, V, W, R = best_solution
        # Convert to lists of lists as required
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
