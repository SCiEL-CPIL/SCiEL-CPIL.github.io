import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
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
                
    start_time = time.time()
    time_limit = 100
    
    def check(U, V, W, R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def round_half(x):
        return np.round(x * 2) / 2

    # Try to find a lower rank
    # We use a heuristic: initialize with random vectors, run a few ALS steps, round, check.
    # If it fails, try next rank.
    
    for R in [80, 70, 60, 50, 45, 40]:
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random restarts
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
            np.random.seed(100 + R + restart)
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            # Simple ALS: Update U, then V, then W
            # Objective: Minimize sum (1 - sum_r ...)^2
            # This is non-convex.
            # We will just do a few iterations of gradient descent on the loss.
            # Loss L = sum_{i,j,k} (1 - S_{i,j,k})^2
            # S_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
            
            # Gradient w.r.t U_{i,j,r}:
            # dL/dU_{i,j,r} = -2 sum_{i,j,k} (1 - S_{i,j,k}) V_{j,k,r} W_{k,i,r}
            # Note: U_{i,j,r} only appears in S_{i,j,k} for fixed i,j.
            # So dL/dU_{i,j,r} = -2 (1 - S_{i,j,k}) V_{j,k,r} W_{k,i,r} ... wait, sum over k?
            # S_{i,j,k} depends on U_{i,j,r}.
            # So for a fixed i,j, U_{i,j,r} affects all S_{i,j,k} for k in 0..p-1.
            # dL/dU_{i,j,r} = -2 sum_{k} (1 - S_{i,j,k}) V_{j,k,r} W_{k,i,r}
            
            lr = 0.01
            for it in range(20):
                # Update U
                for i in range(n):
                    for j in range(m):
                        idx_u = i * m + j
                        grad_u = np.zeros(R)
                        for k in range(p):
                            idx_v = j * p + k
                            idx_w = k * n + i
                            S = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                            grad_u += (1 - S) * V[idx_v, :] * W[idx_w, :]
                        U[idx_u, :] += lr * grad_u
                        
                # Update V
                for j in range(m):
                    for k in range(p):
                        idx_v = j * p + k
                        grad_v = np.zeros(R)
                        for i in range(n):
                            idx_u = i * m + j
                            idx_w = k * n + i
                            S = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                            grad_v += (1 - S) * U[idx_u, :] * W[idx_w, :]
                        V[idx_v, :] += lr * grad_v
                        
                # Update W
                for k in range(p):
                    for i in range(n):
                        idx_w = k * n + i
                        grad_w = np.zeros(R)
                        for j in range(m):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            S = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                            grad_w += (1 - S) * U[idx_u, :] * V[idx_v, :]
                        W[idx_w, :] += lr * grad_w

            # Round and check
            U_r = round_half(U)
            V_r = round_half(V)
            W_r = round_half(W)
            
            if check(U_r, V_r, W_r, R):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }

    # Fallback
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
