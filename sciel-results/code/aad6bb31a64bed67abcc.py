import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_rank = n * m * p
    best_solution = None
    
    # Schoolbook solution
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_solution = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank
    # Target ranks to try
    target_ranks = [24, 26, 28, 30, 32, 34]
    
    for R in target_ranks:
        if time.time() > deadline - 20:
            break
            
        # Try a few random starts
        found = False
        for trial in range(5):
            if time.time() > deadline - 10:
                break
                
            # Initialize
            # Random initialization with small values
            U_init = np.random.randn(n * m, R) * 0.1
            V_init = np.random.randn(m * p, R) * 0.1
            W_init = np.random.randn(p * n, R) * 0.1
            
            # Flatten
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residuals(x):
                U_r = x[:n*m*R].reshape(n*m, R)
                V_r = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_r = x[n*m*R + m*p*R:].reshape(p*n, R)
                
                res = []
                # Constraint: sum_r W[k,i,r] * U[i,j,r] * V[j',k,r] = delta_{j,j'}
                for i in range(n):
                    for k in range(p):
                        # Matrix of size m x m
                        # Entry (j, j') is sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r]
                        M = np.zeros((m, m))
                        for r in range(R):
                            w_val = W_r[k * n + i, r]
                            if w_val != 0:
                                u_vec = U_r[i * m : i * m + m, r] # shape (m,)
                                v_vec = V_r[:, k, r] if V_r.ndim > 2 else V_r.reshape(m, p)[:, k] # Wait, V is (m*p, R)
                                # V index is j*p + k. So for fixed k, we need V[j*p+k, r] for all j.
                                # V_r is (m*p, R).
                                # v_vec[j] = V_r[j*p + k, r]
                                v_vec = np.array([V_r[j * p + k, r] for j in range(m)])
                                M += w_val * np.outer(u_vec, v_vec)
                        
                        # Target is Identity
                        target = np.eye(m)
                        res.append((M - target).flatten())
                        
                return np.concatenate(res)
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=10000)
                if result.success or result.cost < 1e-8:
                    x_sol = result.x
                    U_sol = x_sol[:n*m*R].reshape(n*m, R)
                    V_sol = x_sol[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = x_sol[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    # Check if close to half-integers
                    # Snap to nearest 0.5
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify snapped solution
                    if _verify_solution(U_snap, V_snap, W_snap, n, m, p, R):
                        # Found a valid solution
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                        
            except Exception:
                pass
                
    # Return best found (schoolbook)
    return {
        "U": best_solution["U"].tolist(),
        "V": best_solution["V"].tolist(),
        "W": best_solution["W"].tolist()
    }

def _verify_solution(U, V, W, n, m, p, R):
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m))
            for r in range(R):
                w_val = W[k * n + i, r]
                if w_val != 0:
                    u_vec = U[i * m : i * m + m, r]
                    v_vec = np.array([V[j * p + k, r] for j in range(m)])
                    M += w_val * np.outer(u_vec, v_vec)
            if not np.allclose(M, np.eye(m), atol=1e-6):
                return False
    return True
