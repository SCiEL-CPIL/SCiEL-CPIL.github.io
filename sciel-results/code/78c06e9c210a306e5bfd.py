import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition for fallback
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
                
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # We iterate from schoolbook_rank - 1 down to 50
    # We try to find a decomposition for ranks less than 72
    for r in range(schoolbook_rank - 1, 49, -1):
        if time.time() > deadline:
            break
            
        inits = []
        
        # 1. Truncated schoolbook (often a good starting point for near-rank reductions)
        U_init = U_school[:, :r].copy()
        V_init = V_school[:, :r].copy()
        W_init = W_school[:, :r].copy()
        inits.append((U_init, V_init, W_init))
        
        # 2. Random initializations
        # We use a few random tries to escape local minima
        for _ in range(5):
            U_rand = np.random.randn(size_u, r)
            V_rand = np.random.randn(size_v, r)
            W_rand = np.random.randn(size_w, r)
            # Normalize to prevent numerical issues
            U_rand /= np.linalg.norm(U_rand) * 10
            V_rand /= np.linalg.norm(V_rand) * 10
            W_rand /= np.linalg.norm(W_rand) * 10
            inits.append((U_rand, V_rand, W_rand))
            
        for U_init, V_init, W_init in inits:
            if time.time() > deadline:
                break
                
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residuals(params):
                U = params[0:size_u*r].reshape(n, m, r)
                V = params[size_u*r:size_u*r + size_v*r].reshape(m, p, r)
                W = params[size_u*r + size_v*r:].reshape(p, n, r)
                
                # Compute the tensor D_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
                # Note: W is indexed by k*n+i in the flattened form, so reshape to (p, n, r)
                # The einsum 'ijr,jkr,kir->ijk' computes:
                # U[i,j,r] * V[j,k,r] * W[k,i,r]
                D = np.einsum('ijr,jkr,kir->ijk', U, V, W)
                
                return (D - 1.0).flatten()
            
            try:
                # Use 'lm' (Levenberg-Marquardt) which is fast for non-linear least squares
                # max_nfev is set to allow some exploration but not too much
                result = least_squares(residuals, params0, method='lm', max_nfev=300, ftol=1e-12, xtol=1e-12)
                
                # Check if we found a solution
                if result.success and result.cost < 1e-10:
                    U_sol = result.x[0:size_u*r].reshape(n, m, r)
                    V_sol = result.x[size_u*r:size_u*r + size_v*r].reshape(m, p, r)
                    W_sol = result.x[size_u*r + size_v*r:].reshape(p, n, r)
                    
                    # Snap to nearest half-integer (or integer)
                    # The problem asks for integer/half-integer entries
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify the snapped solution
                    D_check = np.einsum('ijr,jkr,kir->ijk', U_snap, V_snap, W_snap)
                    
                    if np.allclose(D_check, 1.0, atol=1e-9):
                        # Convert to required format: (n*m, r), (m*p, r), (p*n, r)
                        U_out = U_snap.reshape(size_u, r).tolist()
                        V_out = V_snap.reshape(size_v, r).tolist()
                        W_out = W_snap.reshape(size_w, r).tolist()
                        
                        return {
                            "U": U_out,
                            "V": V_out,
                            "W": W_out
                        }
                        
            except Exception:
                pass
                
    # Fallback
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
