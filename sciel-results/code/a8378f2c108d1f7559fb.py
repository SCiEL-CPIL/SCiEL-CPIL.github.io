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
    time_limit = 145  # seconds - safety margin
    
    def try_round_and_verify(U, V, W, T, tol=1e-8):
        """Try rounding to half-integers and verify exact decomposition."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        T_hat_r = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
        E_r = T - T_hat_r
        error_r = np.sum(E_r**2)
        
        if error_r < tol:
            return True, U_r, V_r, W_r
        return False, None, None, None
    
    def polish(U, V, W, T, lr_init=0.01, max_steps=5000, tol=1e-10):
        """High-precision polish pass with adaptive learning rate."""
        lr = lr_init
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
                
            T_hat = np.einsum('ur,vr,wr->uvw', U, V, W)
            E = T - T_hat
            error = np.sum(E**2)
            
            if error < tol:
                return U, V, W, True
            
            dU = -np.einsum('uvw,vr,wr->ur', E, V, W)
            dV = -np.einsum('uvw,ur,wr->vr', E, U, W)
            dW = -np.einsum('uvw,ur,vr->wr', E, U, V)
            
            vU = momentum * vU + lr * dU
            vV = momentum * vV + lr * dV
            vW = momentum * vW + lr * dW
            
            U = U + vU
            V = V + vV
            W = W + vW
            
            if step % 200 == 0:
                lr *= 0.92
        
        return U, V, W, False
    
    # Try ranks from 54 down to 40
    for R in range(54, 39, -1):
        if time.time() - start_time > time_limit:
            break
        
        max_restarts = 4 if R >= 50 else 6
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Strategy 0: Small random
            if restart == 0:
                scale = 0.1
                U = np.random.randn(U_dim, R) * scale
                V = np.random.randn(V_dim, R) * scale
                W = np.random.randn(W_dim, R) * scale
            elif restart == 1:
                # Strategy 1: Medium random
                scale = 0.3
                U = np.random.randn(U_dim, R) * scale
                V = np.random.randn(V_dim, R) * scale
                W = np.random.randn(W_dim, R) * scale
            elif restart == 2:
                # Strategy 2: Large random
                scale = 0.8
                U = np.random.randn(U_dim, R) * scale
                V = np.random.randn(V_dim, R) * scale
                W = np.random.randn(W_dim, R) * scale
            elif restart == 3:
                # Strategy 3: Very small
                scale = 0.05
                U = np.random.randn(U_dim, R) * scale
                V = np.random.randn(V_dim, R) * scale
                W = np.random.randn(W_dim, R) * scale
            elif restart == 4:
                # Strategy 4: Structured - based on schoolbook perturbation
                if R <= rank_schoolbook:
                    U = U_school[:, :R].copy() + np.random.randn(U_dim, R) * 0.1
                    V = V_school[:, :R].copy() + np.random.randn(V_dim, R) * 0.1
                    W = W_school[:, :R].copy() + np.random.randn(W_dim, R) * 0.1
                else:
                    scale = 0.2
                    U = np.random.randn(U_dim, R) * scale
                    V = np.random.randn(V_dim, R) * scale
                    W = np.random.randn(W_dim, R) * scale
            else:
                # Strategy 5: Alternating scales
                scale = 0.15 + 0.1 * restart
                U = np.random.randn(U_dim, R) * scale
                V = np.random.randn(V_dim, R) * scale
                W = np.random.randn(W_dim, R) * scale
            
            # Main gradient descent phase
            lr = 0.03
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            found = False
            for step in range(3000):
                if time.time() - start_time > time_limit:
                    break
                
                T_hat = np.einsum('ur,vr,wr->uvw', U, V, W)
                E = T - T_hat
                error = np.sum(E**2)
                
                if error < 1e-10:
                    success, U_r, V_r, W_r = try_round_and_verify(U, V, W, T)
                    if success:
                        best_rank = R
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        found = True
                    break
                
                dU = -np.einsum('uvw,vr,wr->ur', E, V, W)
                dV = -np.einsum('uvw,ur,wr->vr', E, U, W)
                dW = -np.einsum('uvw,ur,vr->wr', E, U, V)
                
                vU = momentum * vU + lr * dU
                vV = momentum * vV + lr * dV
                vW = momentum * vW + lr * dW
                
                U += vU
                V += vV
                W += vW
                
                if step % 150 == 0:
                    lr *= 0.94
            
            if found:
                break
            
            # If not found but error is low, try polish
            T_hat = np.einsum('ur,vr,wr->uvw', U, V, W)
            error = np.sum((T - T_hat)**2)
            if error < 1e-6:
                U, V, W, converged = polish(U, V, W, T, lr_init=0.005, max_steps=3000)
                if converged:
                    success, U_r, V_r, W_r = try_round_and_verify(U, V, W, T)
                    if success:
                        best_rank = R
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
