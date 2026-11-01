import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    n, m, p = 2, 3, 3
    U_rows = n * m  # 6
    V_rows = m * p  # 9
    W_rows = p * n  # 6
    
    deadline = time.time() + 110  # Safety margin

    # 1. Schoolbook Rank 18
    best_rank = n * m * p
    best_U = np.zeros((U_rows, best_rank))
    best_V = np.zeros((V_rows, best_rank))
    best_W = np.zeros((W_rows, best_rank))
    for c, (i, j, k) in enumerate(np.ndindex(n, m, p)):
        best_U[i * m + j, c] = 1
        best_V[j * p + k, c] = 1
        best_W[k * n + i, c] = 1

    def verify(U, V, W, tol=1e-6):
        r = U.shape[1]
        # Check if sum_t W[k*n+i, t] U[i*m+j, t] V[j'*p+k, t] == delta_{j,j'} for all i,k,j,j'
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, m))
                for t in range(r):
                    w_val = W[k * n + i, t]
                    if w_val == 0: continue
                    u_col = U[:, t][i * m : i * m + m]
                    v_col = V[:, t][k * p : k * p + p] # Wait, V index is j*p+k. So for fixed k, we need V[j*p+k, t] for all j.
                    # V is (m*p, r). Row index is j*p+k.
                    # So for fixed k, the rows are k, p+k, 2p+k.
                    v_col = np.array([V[j * p + k, t] for j in range(m)])
                    M += w_val * np.outer(u_col, v_col)
                if not np.allclose(M, np.eye(m), atol=tol, rtol=tol):
                    return False
        return True

    def snap_to_halfint(arr, tol=1e-4):
        result = arr.copy()
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                val = result[i, j]
                # Snap to nearest integer or half-integer
                candidates = [round(val), round(val * 2) / 2]
                dists = [abs(val - c) for c in candidates]
                min_dist = min(dists)
                if min_dist < tol:
                    result[i, j] = candidates[dists.index(min_dist)]
                else:
                    return None
        return result

    # 2. Search for lower rank (12, 13, 14, 15, 16, 17)
    for r in range(12, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt)
            # Initialize
            U = np.random.randn(U_rows, r) * 0.1
            V = np.random.randn(V_rows, r) * 0.1
            W = np.random.randn(W_rows, r) * 0.1
            
            def objective(params):
                U_p = params[:U_rows * r].reshape(U_rows, r)
                V_p = params[U_rows * r : U_rows * r + V_rows * r].reshape(V_rows, r)
                W_p = params[U_rows * r + V_rows * r :].reshape(W_rows, r)
                
                residual = 0.0
                for i in range(n):
                    for k in range(p):
                        M = np.zeros((m, m))
                        for t in range(r):
                            w_val = W_p[k * n + i, t]
                            if w_val == 0: continue
                            u_col = U_p[i * m : i * m + m, t]
                            v_col = np.array([V_p[j * p + k, t] for j in range(m)])
                            M += w_val * np.outer(u_col, v_col)
                        residual += np.sum((M - np.eye(m)) ** 2)
                return residual

            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B or Nelder-Mead
            # L-BFGS-B is faster for smooth problems
            try:
                res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8})
                if res.fun < 1e-4:
                    # Reconstruct
                    U_sol = res.x[:U_rows * r].reshape(U_rows, r)
                    V_sol = res.x[U_rows * r : U_rows * r + V_rows * r].reshape(V_rows, r)
                    W_sol = res.x[U_rows * r + V_rows * r :].reshape(W_rows, r)
                    
                    # Try to snap to half-integers
                    U_snap = snap_to_halfint(U_sol)
                    if U_snap is not None:
                        V_snap = snap_to_halfint(V_sol)
                        if V_snap is not None:
                            W_snap = snap_to_halfint(W_sol)
                            if W_snap is not None:
                                if verify(U_snap, V_snap, W_snap):
                                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                                    best_rank = r
                                    found = True
                                    break
            except Exception:
                pass

        if found:
            break

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
