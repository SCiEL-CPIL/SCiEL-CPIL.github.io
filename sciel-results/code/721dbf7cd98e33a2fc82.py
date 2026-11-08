import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 23
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Target tensor T[i, j, k] = 1 if i == k else 0
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == k:
                    T[i, j, k] = 1.0

    def residual(params, R):
        # params: flattened U (n*m*R), V (m*p*R), W (p*n*R)
        U = params[:n*m*R].reshape(n*m, R)
        V = params[n*m*R:2*n*m*R].reshape(m*p, R)
        W = params[2*n*m*R:].reshape(p*n, R)
        
        # Compute tensor approximation
        # T_approx[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Vectorized computation
        # U is (n*m, R), V is (m*p, R), W is (p*n, R)
        # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # Reshape U to (n, m, R)
        U_3d = U.reshape(n, m, R)
        # Reshape V to (m, p, R)
        V_3d = V.reshape(m, p, R)
        # Reshape W to (p, n, R)
        W_3d = W.reshape(p, n, R)
        
        # Sum over r
        # T_approx[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
        
        # Efficient way:
        # For each r, compute outer product? No, that's R times.
        # Use einsum
        # 'ijr,jkr,kir->ijk'
        T_approx = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        return np.sum((T_approx - T)**2)

    def try_rank(R, max_restarts=20):
        current_time = time.time()
        if current_time > deadline:
            return None
            
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            # Scale initialization to help optimization
            scale = 1.0
            params = np.random.normal(0, scale, size=3 * n * m * R)
            
            # Optimization
            # Use L-BFGS-B or BFGS
            # We need to be careful with the number of iterations
            try:
                res = minimize(
                    residual, 
                    params, 
                    args=(R,), 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-12, 'maxfun': 1000}
                )
            except Exception:
                continue
                
            if res.fun < 1e-6:
                # Check if we can round to half-integers
                U_flat = res.x[:n*m*R]
                V_flat = res.x[n*m*R:2*n*m*R]
                W_flat = res.x[2*n*m*R:]
                
                # Snap to nearest half-integer
                U_snap = np.round(U_flat * 2) / 2
                V_snap = np.round(V_flat * 2) / 2
                W_snap = np.round(W_flat * 2) / 2
                
                # Verify snapped solution
                params_snap = np.concatenate([U_snap, V_snap, W_snap])
                res_snap = residual(params_snap, R)
                
                if res_snap < 1e-6:
                    # Valid half-integer solution
                    U = U_snap.reshape(n*m, R)
                    V = V_snap.reshape(m*p, R)
                    W = W_snap.reshape(p*n, R)
                    
                    # Convert to lists
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
            
            # If not exact, maybe try a few more iterations?
            # Or just discard.

    # Try ranks from 23 upwards
    for R in range(23, 28):
        result = try_rank(R)
        if result:
            return result
            
    # Fallback: Schoolbook
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
