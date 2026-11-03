"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    def compute_residuals(x, R):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U = x[0:size_u*R].reshape(size_u, R)
        V = x[size_u*R:(size_u+size_v)*R].reshape(size_v, R)
        W = x[(size_u+size_v)*R:].reshape(size_w, R)
        
        residuals = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m + j]
                    v_val = V[j*p + k]
                    w_val = W[k*n + i]
                    s = np.dot(u_val, v_val * w_val)
                    residuals[idx] = s - 1.0
                    idx += 1
        return residuals

    def snap_and_verify(U, V, W, step):
        U_snap = np.round(U / step) * step
        V_snap = np.round(V / step) * step
        W_snap = np.round(W / step) * step
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_snap[i*m + j]
                    v_val = V_snap[j*p + k]
                    w_val = W_snap[k*n + i]
                    s = np.dot(u_val, v_val * w_val)
                    if not np.isclose(s, 1.0, atol=1e-9):
                        return None
        return {"U": U_snap, "V": V_snap, "W": W_snap}

    # Fallback schoolbook
    def get_schoolbook():
        R = n * m * p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_solution = None
    best_rank = n * m * p
    
    # Try ranks from 55 upwards
    for R in range(55, best_rank + 1):
        if time.time() > deadline - 15:
            break
            
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_vars = (size_u + size_v + size_w) * R
        
        num_restarts = 10 if R <= 56 else 5
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            x0 = np.random.normal(0, 1, total_vars)
            
            try:
                result = least_squares(compute_residuals, x0, args=(R,), method='trf', max_nfev=2000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                
                if result.cost < 1e-8:
                    U = result.x[0:size_u*R].reshape(size_u, R)
                    V = result.x[size_u*R:(size_u+size_v)*R].reshape(size_v, R)
                    W = result.x[(size_u+size_v)*R:].reshape(size_w, R)
                    
                    # Try snapping to 0.5
                    sol = snap_and_verify(U, V, W, 0.5)
                    if sol:
                        best_solution = sol
                        best_rank = R
                        if R == 55:
                            return best_solution
                        break
                    
                    # Try snapping to 1
                    sol = snap_and_verify(U, V, W, 1.0)
                    if sol:
                        best_solution = sol
                        best_rank = R
                        if R == 55:
                            return best_solution
                        break
            except Exception:
                pass

    if best_solution is None:
        best_solution = get_schoolbook()
        
    return best_solution
# EVOLVE-BLOCK-END
