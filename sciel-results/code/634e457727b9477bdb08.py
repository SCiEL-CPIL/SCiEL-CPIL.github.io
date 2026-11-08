import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((U_dim, rank_schoolbook))
    V_school = np.zeros((V_dim, rank_schoolbook))
    W_school = np.zeros((W_dim, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_school[u, col] = 1
                V_school[v, col] = 1
                W_school[w, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    start_time = time.time()
    time_limit = 115  # seconds
    
    # Try ranks from 53 down to 30
    # We will try a few restarts for each rank
    # To save time, we can skip some ranks if they fail quickly?
    # Or prioritize certain ranks?
    # Let's just go down.
    
    for R in range(53, 29, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Allow more restarts for lower ranks? No, lower ranks are harder.
        # Allow fewer restarts for high ranks (easier).
        # Let's do 3 restarts for R > 45, 5 for R <= 45.
        max_restarts = 3 if R > 45 else 5
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Initialization
            # Try different scales
            scale = 0.1
            U = np.random.randn(U_dim, R) * scale
            V = np.random.randn(V_dim, R) * scale
            W = np.random.randn(W_dim, R) * scale
            
            # Gradient Descent
            lr = 0.05
            for step in range(1000):
                if time.time() - start_time > time_limit:
                    break
                
                T_hat = np.einsum('ur,vr,wr->uvw', U, V, W)
                E = T - T_hat
                error = np.sum(E**2)
                
                if error < 1e-10:
                    # Try rounding
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    T_hat_r = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
                    E_r = T - T_hat_r
                    error_r = np.sum(E_r**2)
                    
                    if error_r < 1e-8:
                        best_rank = R
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                    else:
                        # Keep float if rounding fails? 
                        # The problem asks for integer/half-integer.
                        # If we can't satisfy that, maybe we shouldn't return this rank?
                        # But if we return float, it might be accepted if tolerance is loose.
                        # Let's assume we need to satisfy the "integer/half-integer" constraint strictly.
                        # So we only accept if rounding works.
                        pass
                    
                    break
                
                dU = np.einsum('uvw,vr,wr->ur', E, V, W)
                dV = np.einsum('uvw,ur,wr->vr', E, U, W)
                dW = np.einsum('uvw,ur,vr->wr', E, U, V)
                
                U -= 2 * lr * dU
                V -= 2 * lr * dV
                W -= 2 * lr * dW
                
                if step % 100 == 0:
                    lr *= 0.95
            
            if best_rank == R:
                break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
