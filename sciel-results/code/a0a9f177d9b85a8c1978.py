import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        """Verify tensor decomposition exactly."""
        # Check half-integer constraint
        if not np.allclose(U * 2, np.round(U * 2), atol=tol):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=tol):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=tol):
            return False
        
        R = U.shape[1]
        U_mat = U.reshape(n, m, R)
        V_mat = V.reshape(m, p, R)
        W_mat = W.reshape(p, n, R)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    uv = U_mat[i, j, :] * V_mat[j, k, :]
                    # T[ip, kp] = sum_r uv[r] * W_mat[kp, ip, r]
                    # W_mat shape is (p, n, R), so W_mat[kp, ip, r]
                    # We want result[ip, kp] = sum_r uv[r] * W_mat[kp, ip, r]
                    # = (W_mat.T @ uv) where W_mat.T has shape (n, p, R)
                    # Actually: result[ip, kp] = sum_r uv[r] * W_mat[kp, ip, r]
                    # Let's compute: for each ip, kp: sum_r uv[r] * W_mat[kp, ip, r]
                    # = np.einsum('r,kpir->ip', uv, W_mat) but that's not right
                    # W_mat[kp, ip, r], so for fixed ip,kp: sum_r uv[r]*W_mat[kp,ip,r]
                    # This is: (uv * W_mat.T) summed over r -> but W_mat.T is (n,p,R)
                    # W_mat.T[ip, kp, r] = W_mat[kp, ip, r]
                    # So result[ip,kp] = sum_r uv[r] * W_mat.T[ip, kp, r]
                    # = (W_mat.T @ uv) doesn't work directly
                    # result = np.einsum('r,ipr->ip', uv, W_mat.T)
                    T_slice = np.einsum('r,ipr->ip', uv, W_mat.T)  # shape (n, p), T_slice[ip, kp]
                    
                    if abs(T_slice[i, k] - 1.0) > tol:
                        return False
                    mask = np.ones((n, p), dtype=bool)
                    mask[i, k] = False
                    if np.any(np.abs(T_slice[mask]) > tol):
                        return False
        return True
    
    def residual(x):
        """Compute residual for optimization."""
        total_size = n*m + m*p + p*n
        R = len(x) // total_size
        if R == 0:
            return np.array([])
        
        U = x[:n*m*R].reshape(n, m, R)
        V = x[n*m*R : n*m*R + m*p*R].reshape(m, p, R)
        W = x[n*m*R + m*p*R :].reshape(p, n, R)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    uv = U[i, j, :] * V[j, k, :]
                    T_slice = np.einsum('r,ipr->ip', uv, W.T)  # shape (n, p)
                    
                    # Diagonal: T_slice[i, k] should be 1
                    res.append(T_slice[i, k] - 1.0)
                    # Off-diagonal should be 0
                    mask = np.ones((n, p), dtype=bool)
                    mask[i, k] = False
                    res.extend(T_slice[mask].tolist())
        
        return np.array(res)
    
    best_solution = (U_school.tolist(), V_school.tolist(), W_school.tolist())
    best_rank = rank_schoolbook
    
    # Strategy 1: Try to reduce rank by finding linear dependencies in schoolbook
    # Strategy 2: Random restarts with optimization
    # Strategy 3: Greedy column elimination
    
    # First, try greedy elimination from schoolbook
    U_curr = U_school.copy()
    V_curr = V_school.copy()
    W_curr = W_school.copy()
    R_curr = rank_schoolbook
    
    # Try to eliminate columns by checking if one column is linearly dependent
    # This is complex, so let's focus on optimization-based approach
    
    # Optimization-based search
    for R in range(rank_schoolbook - 1, 50, -1):
        if time.time() > deadline:
            break
        
        total_size = n*m + m*p + p*n
        found = False
        
        # Multiple random restarts with different strategies
        for restart in range(8):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + R * 1000 + restart * 7)
            
            # Different initialization strategies
            if restart < 2:
                # Small random
                x0 = np.random.randn(total_size * R) * 0.1
            elif restart < 4:
                # Larger random
                x0 = np.random.randn(total_size * R) * 0.5
            elif restart < 6:
                # Sparse random
                x0 = np.random.randn(total_size * R)
                x0[np.abs(x0) < 0.8] = 0
            else:
                # Mixed
                x0 = np.random.randn(total_size * R) * 0.3
                x0[::3] = np.random.randn(len(x0[::3])) * 0.7
            
            try:
                sol = least_squares(residual, x0, method='lm', max_nfev=5000, ftol=1e-12, xtol=1e-12)
                if sol.cost < 1e-8:
                    x_opt = sol.x
                    U_opt = x_opt[:n*m*R].reshape(n, m, R)
                    V_opt = x_opt[n*m*R : n*m*R + m*p*R].reshape(m, p, R)
                    W_opt = x_opt[n*m*R + m*p*R :].reshape(p, n, R)
                    
                    # Round to nearest 0.5
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if verify(U_round, V_round, W_round, tol=1e-9):
                        best_solution = (U_round.flatten().tolist(), V_round.flatten().tolist(), W_round.flatten().tolist())
                        best_rank = R
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break  # Found a lower rank, stop
    
    U_final, V_final, W_final = best_solution
    return {"U": U_final, "V": V_final, "W": W_final}
