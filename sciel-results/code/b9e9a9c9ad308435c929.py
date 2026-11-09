import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    T_target = np.ones((n, m, p))
    
    def compute_error(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        err = np.sum((T - T_target)**2)
        return err

    def round_to_half_int(A):
        # Round to nearest 0.5
        return np.round(A * 2) / 2.0

    def local_search_polish(U, V, W, max_steps=2000):
        # Simulated Annealing / Local Search on discrete half-integers
        # to fix small errors introduced by rounding
        values = np.array([-1, -0.5, 0, 0.5, 1])
        rng = np.random.default_rng(42)
        
        err = compute_error(U, V, W)
        if err < 1e-9:
            return U, V, W, err
            
        current_temp = 1.0
        min_temp = 1e-6
        cooling = 0.95
        
        rank = U.shape[1]
        
        for step in range(max_steps):
            if time.time() > deadline - 1.0:
                break
                
            if current_temp < min_temp:
                break
                
            # Pick a random matrix and a random entry
            mat_idx = rng.integers(0, 3)
            if mat_idx == 0:
                mat = U
                size = n*m
            elif mat_idx == 1:
                mat = V
                size = m*p
            else:
                mat = W
                size = p*n
                
            idx = rng.integers(0, size)
            r_idx = rng.integers(0, rank)
            
            old_val = mat[idx, r_idx]
            new_val = rng.choice(values)
            
            if new_val == old_val:
                continue
                
            mat[idx, r_idx] = new_val
            new_err = compute_error(U, V, W)
            
            delta = new_err - err
            if delta < 0 or rng.random() < np.exp(-delta / current_temp):
                err = new_err
                if err < 1e-9:
                    return U, V, W, err
            else:
                mat[idx, r_idx] = old_val
                
            current_temp *= cooling
            
        return U, V, W, err

    # Try ranks from lower bounds upwards
    # Known lower bound for 2x3x5 is likely around 20-23.
    # Schoolbook is 30.
    for rank in range(18, 31):
        if time.time() > deadline - 5.0:
            break
            
        # Multiple restarts with ALS
        for attempt in range(20):
            if time.time() > deadline - 2.0:
                break
                
            rng = np.random.default_rng(42 + attempt)
            
            # Random initialization
            U = rng.standard_normal((n*m, rank))
            V = rng.standard_normal((m*p, rank))
            W = rng.standard_normal((p*n, rank))
            
            # ALS Iterations to minimize continuous error
            # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            # We solve for one matrix at a time, keeping others fixed.
            
            for it in range(50):
                if time.time() > deadline - 1.0:
                    break
                    
                # Update U
                # For each (i, j), U[i*m+j, :] is a vector u
                # T[i, j, :] = u @ (V[j, :, :] * W[:, i, :]).T ?
                # Let's derive:
                # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # Let A_k = V[j*p+k, :] * W[k*n+i, :] (element-wise product, length rank)
                # Then T[i, j, k] = sum_c U[i*m+j, c] A_k[c] = dot(U[i*m+j, :], A_k)
                # So for fixed i, j, we have a system of p equations:
                # U[i*m+j, :] @ A_k^T = T[i, j, k] for k=0..p-1
                # Let M_U (p x rank) where M_U[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Then U[i*m+j, :] @ M_U.T = T[i, j, :]
                # U[i*m+j, :] = T[i, j, :] @ inv(M_U.T)  (if square/invertible)
                # Or least squares: U[i*m+j, :] = lstsq(M_U, T[i, j, :])
                
                for i in range(n):
                    for j in range(m):
                        M_U = np.zeros((p, rank))
                        for k in range(p):
                            M_U[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b = T_target[i, j, :]
                        # Solve M_U @ u = b  => u is column vector of size rank
                        # M_U (p x rank) * u (rank x 1) = b (p x 1)
                        u, _, _, _ = np.linalg.lstsq(M_U, b, rcond=None)
                        U[i*m+j, :] = u.flatten()
                        
                # Update V
                # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # For fixed j, k:
                # Let N_j = U[:, j, :] * W[k, :, :] (element-wise, n x rank)
                # T[:, j, k] = N_j @ V[j*p+k, :].T
                # V[j*p+k, :] = T[:, j, k] @ inv(N_j.T)
                
                for j in range(m):
                    for k in range(p):
                        N_V = np.zeros((n, rank))
                        for i in range(n):
                            N_V[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b = T_target[:, j, k]
                        # Solve N_V @ v = b
                        v, _, _, _ = np.linalg.lstsq(N_V, b, rcond=None)
                        V[j*p+k, :] = v.flatten()

                # Update W
                # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # For fixed k, i:
                # Let P_k = U[i, :, :] * V[:, k, :] (element-wise, m x rank)
                # T[i, :, k] = P_k @ W[k*n+i, :].T
                # W[k*n+i, :] = T[i, :, k] @ inv(P_k.T)
                
                for k in range(p):
                    for i in range(n):
                        P_W = np.zeros((m, rank))
                        for j in range(m):
                            P_W[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b = T_target[i, :, k]
                        # Solve P_W @ w = b
                        w, _, _, _ = np.linalg.lstsq(P_W, b, rcond=None)
                        W[k*n+i, :] = w.flatten()
            
            # After ALS, round to half-integers
            U_rounded = round_to_half_int(U)
            V_rounded = round_to_half_int(V)
            W_rounded = round_to_half_int(W)
            
            err = compute_error(U_rounded, V_rounded, W_rounded)
            
            if err < 1e-9:
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
            
            # If not exact, try to polish with local search
            U_polished, V_polished, W_polished, err_polished = local_search_polish(
                U_rounded.copy(), V_rounded.copy(), W_rounded.copy(), max_steps=1000
            )
            
            if err_polished < 1e-9:
                return {
                    "U": U_polished.tolist(),
                    "V": V_polished.tolist(),
                    "W": W_polished.tolist()
                }
            
            # Also try rounding the original continuous solution to integers only? 
            # Or just keep the best continuous solution if it's very close?
            # The task requires integer/half-integer entries.
            
    # If no better solution found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
