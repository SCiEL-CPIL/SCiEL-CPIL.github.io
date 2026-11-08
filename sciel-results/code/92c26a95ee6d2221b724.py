import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110  # Safety margin before 120s limit

    def error_np(U, V, W):
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err

    # Schoolbook decomposition (rank 63)
    R0 = n * m * p
    U0 = np.zeros((n*m, R0))
    V0 = np.zeros((m*p, R0))
    W0 = np.zeros((p*n, R0))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, c] = 1
                V0[j*p+k, c] = 1
                W0[k*n+i, c] = 1
                c += 1

    best_R = R0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    def snap_half(x):
        return round(x * 2) / 2

    def local_search(U, V, W, max_iters=300):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        best_local_err = error_np(U, V, W)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            err = error_np(U, V, W)
            if err < 1e-9:
                return U, V, W
            
            # Random perturbation
            choice = random.random()
            if choice < 0.33:
                idx = random.randint(0, n*m-1)
                col = random.randint(0, r-1)
                U[idx, col] += random.choice([-1, 1]) * 0.5
                U[idx, col] = snap_half(U[idx, col])
            elif choice < 0.66:
                idx = random.randint(0, m*p-1)
                col = random.randint(0, r-1)
                V[idx, col] += random.choice([-1, 1]) * 0.5
                V[idx, col] = snap_half(V[idx, col])
            else:
                idx = random.randint(0, p*n-1)
                col = random.randint(0, r-1)
                W[idx, col] += random.choice([-1, 1]) * 0.5
                W[idx, col] = snap_half(W[idx, col])
            
            new_err = error_np(U, V, W)
            if new_err < best_local_err:
                best_local_err = new_err
            else:
                # Revert
                if choice < 0.33:
                    U[idx, col] = snap_half(U[idx, col] - random.choice([-1, 1]) * 0.5)
                elif choice < 0.66:
                    V[idx, col] = snap_half(V[idx, col] - random.choice([-1, 1]) * 0.5)
                else:
                    W[idx, col] = snap_half(W[idx, col] - random.choice([-1, 1]) * 0.5)
        
        return U, V, W

    # Strategy 1: Try to zero out columns from schoolbook
    U_temp = U0.copy()
    V_temp = V0.copy()
    W_temp = W0.copy()
    current_R = R0
    
    for col in range(R0 - 1, -1, -1):
        if time.time() > deadline - 10:
            break
        
        # Try zeroing this column
        U_bak = U_temp.copy()
        V_bak = V_temp.copy()
        W_bak = W_temp.copy()
        
        U_temp[:, col] = 0
        V_temp[:, col] = 0
        W_temp[:, col] = 0
        
        if error_np(U_temp, V_temp, W_temp) < 1e-9:
            # Keep it zeroed, remove column
            U_temp = np.delete(U_temp, col, axis=1)
            V_temp = np.delete(V_temp, col, axis=1)
            W_temp = np.delete(W_temp, col, axis=1)
            current_R -= 1
        else:
            U_temp, V_temp, W_temp = U_bak, V_bak, W_bak
    
    if current_R < best_R:
        best_R = current_R
        best_U, best_V, best_W = U_temp.copy(), V_temp.copy(), W_temp.copy()

    # Strategy 2: Random search for lower ranks
    for target_R in range(max(15, best_R - 20), best_R):
        if time.time() > deadline - 10:
            break
        
        found = False
        max_restarts = 50
        
        for _ in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            U = U0.copy()
            V = V0.copy()
            W = W0.copy()
            
            if target_R < R0:
                cols_to_zero = random.sample(range(R0), R0 - target_R)
                for c in cols_to_zero:
                    U[:, c] = 0
                    V[:, c] = 0
                    W[:, c] = 0
            
            U, V, W = local_search(U, V, W, max_iters=150)
            
            if error_np(U, V, W) < 1e-9:
                best_R = target_R
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break
        
        if found:
            break

    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
