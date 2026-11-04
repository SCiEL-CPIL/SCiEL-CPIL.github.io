import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Target tensor T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p)
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Precompute indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i, j, k))
                
    # Flattened index mapping
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def calc_error(U, V, W):
        err = 0.0
        for i, j, k in indices:
            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
            err += (val - 1.0)**2
        return err

    def calc_gradient(U, V, W):
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i, j, k in indices:
            idx_u = i*m+j
            idx_v = j*p+k
            idx_w = k*n+i
            val = np.dot(U[idx_u], V[idx_v] * W[idx_w])
            err_term = 2.0 * (val - 1.0)
            
            dU[idx_u] += err_term * (V[idx_v] * W[idx_w])
            dV[idx_v] += err_term * (U[idx_u] * W[idx_w])
            dW[idx_w] += err_term * (U[idx_u] * V[idx_v])
            
        return dU, dV, dW

    def objective(params):
        R = len(params) // (n*m + m*p + p*n)
        U = params[:n*m*R].reshape(n*m, R)
        V = params[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R :].reshape(p*n, R)
        return calc_error(U, V, W)

    def objective_gradient(params):
        R = len(params) // (n*m + m*p + p*n)
        U = params[:n*m*R].reshape(n*m, R)
        V = params[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R :].reshape(p*n, R)
        
        dU, dV, dW = calc_gradient(U, V, W)
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return grad

    best_rank = 36
    # Schoolbook initialization
    U_school = np.zeros((n*m, 36))
    V_school = np.zeros((m*p, 36))
    W_school = np.zeros((p*n, 36))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1

    if time.time() > deadline - 10:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Search for lower rank
    # Lower bound is 27. Try 27 to 35.
    for R in range(27, 36):
        if time.time() > deadline - 5:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(attempt * 100 + R)
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # Combine into a single vector
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B
            try:
                res = opt.minimize(
                    objective, 
                    params, 
                    jac=objective_gradient, 
                    method='L-BFGS-B', 
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000}
                )
            except Exception:
                continue
                
            if res.success and res.fun < 1e-6:
                # Extract U, V, W
                U_sol = res.x[:n*m*R].reshape(n*m, R)
                V_sol = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_sol = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Snap to nearest 0.5
                U_snap = np.round(U_sol * 2.0) / 2.0
                V_snap = np.round(V_sol * 2.0) / 2.0
                W_snap = np.round(W_sol * 2.0) / 2.0
                
                # Check exactness
                if calc_error(U_snap, V_snap, W_snap) < 1e-9:
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                else:
                    # If snapping fails, maybe the solution is not 0.5-based?
                    # But the prompt requires it. 
                    # If we found a valid real solution, we might need to search for a 0.5 solution specifically?
                    # Or maybe the real solution IS 0.5 based but numerical error prevented snapping?
                    # Let's try a tighter snap or check if the real solution works (it might not satisfy the entry constraint).
                    # The prompt says "with integer/half-integer entries".
                    # If the real solution doesn't snap, we can't return it.
                    # We should continue searching or try to refine the snap.
                    pass

    # If no better rank found, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
