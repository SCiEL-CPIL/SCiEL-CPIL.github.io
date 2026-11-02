import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    # Fallback: Schoolbook
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for 120s limit
    
    # Try to find a decomposition for ranks less than 72
    # We iterate from 71 downwards
    for r in range(schoolbook_rank - 1, 35, -1):
        if time.time() > deadline:
            break
            
        # Initialize with the first r columns of the schoolbook decomposition
        # This provides a good starting point close to the solution manifold
        U_init = U_school[:, :r].copy()
        V_init = V_school[:, :r].copy()
        W_init = W_school[:, :r].copy()
        
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        # Flatten for optimization
        params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residuals(params):
            U = params[0:size_u*r].reshape(size_u, r)
            V = params[size_u*r:size_u*r + size_v*r].reshape(size_v, r)
            W = params[size_u*r + size_v*r:].reshape(size_w, r)
            
            D = np.zeros((n, m, p))
            for c in range(r):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                D += Uc[:, :, None] * Vc[None, :, :] * Wc.T[:, None, :]
            
            return (D - 1.0).flatten()
        
        try:
            # Use 'lm' method for unconstrained least squares
            # 'trf' might be better if we add bounds, but 'lm' is faster for small problems
            result = least_squares(residuals, params0, method='lm', max_nfev=2000, ftol=1e-15, xtol=1e-15)
            
            if result.success and result.cost < 1e-10:
                # Check if the solution is close to integer or half-integer
                U_sol = result.x[0:size_u*r].reshape(size_u, r)
                V_sol = result.x[size_u*r:size_u*r + size_v*r].reshape(size_v, r)
                W_sol = result.x[size_u*r + size_v*r:].reshape(size_w, r)
                
                # Snap to nearest half-integer
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                # Verify the snapped solution
                D_check = np.zeros((n, m, p))
                for c in range(r):
                    Uc = U_snap[:, c].reshape(n, m)
                    Vc = V_snap[:, c].reshape(m, p)
                    Wc = W_snap[:, c].reshape(p, n)
                    D_check += Uc[:, :, None] * Vc[None, :, :] * Wc.T[:, None, :]
                
                if np.allclose(D_check, 1.0, atol=1e-9):
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    
        except Exception:
            pass
            
    # If no better rank found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
