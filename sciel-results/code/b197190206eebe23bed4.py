import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Define the target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1
                
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution
    U_school = np.zeros((n*m, n*m*p), dtype=np.int8)
    V_school = np.zeros((m*p, n*m*p), dtype=np.int8)
    W_school = np.zeros((p*n, n*m*p), dtype=np.int8)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    values = [-1, 0, 1]
    
    def try_rank(r, seed):
        np.random.seed(seed)
        U = np.random.choice(values, size=(n*m, r)).astype(np.int8)
        V = np.random.choice(values, size=(m*p, r)).astype(np.int8)
        W = np.random.choice(values, size=(p*n, r)).astype(np.int8)
        
        # Compute initial E and R
        E = np.einsum('ac,bc,gc->abc', U, V, W)
        R = E - T
        current_error = np.sum(R**2)
        
        if current_error == 0:
            return U, V, W
            
        steps = 2000
        for step in range(steps):
            if time.time() > deadline:
                return None
                
            factor = np.random.randint(3)
            if factor == 0:
                i, c = np.random.randint(n*m), np.random.randint(r)
                old_val = U[i, c]
                new_val = np.random.choice(values)
                if new_val == old_val: continue
                delta = new_val - old_val
                
                # Update error
                # R[i, :, :] += delta * V[:, c] * W[:, c]
                v_c = V[:, c]
                w_c = W[:, c]
                update = delta * np.outer(v_c, w_c)
                
                # Calculate change in error
                # ||R + update||^2 - ||R||^2 = 2 * <R, update> + ||update||^2
                # R is 3D, update is 2D (for fixed i)
                r_slice = R[i, :, :]
                dot_prod = np.sum(r_slice * update)
                norm_sq = np.sum(update**2)
                delta_error = 2 * dot_prod + norm_sq
                
                if current_error + delta_error < current_error:
                    U[i, c] = new_val
                    R[i, :, :] += update
                    current_error += delta_error
                    if current_error == 0:
                        return U, V, W
            elif factor == 1:
                i, c = np.random.randint(m*p), np.random.randint(r)
                old_val = V[i, c]
                new_val = np.random.choice(values)
                if new_val == old_val: continue
                delta = new_val - old_val
                
                u_c = U[:, c]
                w_c = W[:, c]
                # Update R[:, i, :]
                # R[:, i, gamma] += delta * U[alpha, c] * W[gamma, c]
                update = delta * np.outer(u_c, w_c)
                
                r_slice = R[:, i, :]
                dot_prod = np.sum(r_slice * update)
                norm_sq = np.sum(update**2)
                delta_error = 2 * dot_prod + norm_sq
                
                if current_error + delta_error < current_error:
                    V[i, c] = new_val
                    R[:, i, :] += update
                    current_error += delta_error
                    if current_error == 0:
                        return U, V, W
            else:
                i, c = np.random.randint(p*n), np.random.randint(r)
                old_val = W[i, c]
                new_val = np.random.choice(values)
                if new_val == old_val: continue
                delta = new_val - old_val
                
                u_c = U[:, c]
                v_c = V[:, c]
                # Update R[:, :, i]
                # R[alpha, beta, i] += delta * U[alpha, c] * V[beta, c]
                update = delta * np.outer(u_c, v_c)
                
                r_slice = R[:, :, i]
                dot_prod = np.sum(r_slice * update)
                norm_sq = np.sum(update**2)
                delta_error = 2 * dot_prod + norm_sq
                
                if current_error + delta_error < current_error:
                    W[i, c] = new_val
                    R[:, :, i] += update
                    current_error += delta_error
                    if current_error == 0:
                        return U, V, W
                        
        return None

    # Try ranks
    for r in range(50, 39, -1):
        if time.time() > deadline:
            break
        for restart in range(10):
            if time.time() > deadline:
                break
            res = try_rank(r, seed=42+restart)
            if res is not None:
                U, V, W = res
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
