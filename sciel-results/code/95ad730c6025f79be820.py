import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
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
    
    def verify_vectorized(U, V, W):
        """Vectorized verification of the tensor decomposition."""
        # Check half-integer constraint
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9):
            return False
        
        R = U.shape[1]
        U_mat = U.reshape(n, m, R)
        V_mat = V.reshape(m, p, R)
        W_mat = W.reshape(p, n, R)
        
        # Compute full tensor: T[i,j,k,ip,kp] = sum_r U[i,j,r]*V[j,k,r]*W[kp,ip,r]
        # We need T[i,j,k,i,k] = 1 and T[i,j,k,ip,kp] = 0 for (ip,kp) != (i,k)
        
        # Use einsum to compute all needed values efficiently
        # T[i,j,k,ip,kp] = sum_r U[i,j,r] * V[j,k,r] * W[kp,ip,r]
        # This is a 5D tensor of shape (n, m, p, n, p)
        
        # For each (i,j,k), compute vector over (ip,kp)
        # T_slice[i,j,k] has shape (n, p) where T_slice[i,j,k][ip,kp] = sum_r U[i,j,r]*V[j,k,r]*W[kp,ip,r]
        
        errors = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute T[i,j,k,ip,kp] for all ip,kp
                    # = sum_r U[i,j,r] * V[j,k,r] * W[kp,ip,r]
                    # = (U[i,j,:] * V[j,k,:]) @ W.reshape(p*n, R).T ... no
                    # Let's do: for each r, contribution is U[i,j,r]*V[j,k,r]*W[:, :, r] reshaped
                    uv = U_mat[i, j, :] * V_mat[j, k, :]  # shape (R,)
                    # T[., ., ., ip, kp] = sum_r uv[r] * W_mat[kp, ip, r]
                    # = uv @ W_mat[:,:,r] summed over r -> shape (p, n) then transpose to (n,p)?
                    # W_mat has shape (p, n, R), so W_mat[kp, ip, r]
                    # result[ip, kp] = sum_r uv[r] * W_mat[kp, ip, r]
                    # = (W_mat.transpose(1, 0, 2) @ uv) -> shape (n, p) where [ip, kp]
                    T_slice = W_mat.transpose(1, 0, 2) @ uv  # shape (n, p), T_slice[ip, kp]
                    
                    # Check T_slice[i, k] == 1
                    if abs(T_slice[i, k] - 1.0) > 1e-9:
                        return False
                    # Check all other entries are 0
                    mask = np.ones((n, p), dtype=bool)
                    mask[i, k] = False
                    if np.any(np.abs(T_slice[mask]) > 1e-9):
                        return False
        return True

    def residual_vectorized(x):
        """Vectorized residual computation."""
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
                    T_slice = W.transpose(1, 0, 2) @ uv  # shape (n, p)
                    
                    # Diagonal: T_slice[i, k] should be 1
                    res.append(T_slice[i, k] - 1.0)
                    # Off-diagonal
                    mask = np.ones((n, p), dtype=bool)
                    mask[i, k] = False
                    res.extend(T_slice[mask].tolist())
        
        return np.array(res)

    best_solution = (U_school.tolist(), V_school.tolist(), W_school.tolist())
    best_rank = rank_schoolbook
    
    # Try to find lower rank with multiple restarts
    for R in range(95, 40, -1):
        if time.time() > deadline:
            break
        
        total_size = n*m + m*p + p*n
        found = False
        
        # Multiple random restarts for each rank
        for restart in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + R * 100 + restart)
            x0 = np.random.randn(total_size * R) * 0.1
            
            try:
                sol = least_squares(residual_vectorized, x0, method='lm', max_nfev=2000)
                if sol.cost < 1e-6:
                    x_opt = sol.x
                    U_opt = x_opt[:n*m*R].reshape(n, m, R)
                    V_opt = x_opt[n*m*R : n*m*R + m*p*R].reshape(m, p, R)
                    W_opt = x_opt[n*m*R + m*p*R :].reshape(p, n, R)
                    
                    # Round to nearest 0.5
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if verify_vectorized(U_round, V_round, W_round):
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
