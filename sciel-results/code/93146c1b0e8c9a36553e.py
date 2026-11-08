"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target multiplication tensor T[u, v, w]
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    T_flat = T.reshape(-1)

    def reconstruct(U, V, W):
        return np.einsum('ur,vr,wr->uvw', U, V, W)

    def error_fn(U, V, W):
        Rec = reconstruct(U, V, W)
        return np.linalg.norm(Rec - T)

    def objective(x):
        # x is flattened [U, V, W]
        n_U = nm * R
        n_V = mp * R
        n_W = pn * R
        
        U = x[:n_U].reshape(nm, R)
        V = x[n_U:n_U+n_V].reshape(mp, R)
        W = x[n_U+n_V:].reshape(pn, R)
        
        Rec = reconstruct(U, V, W)
        diff = Rec - T
        return np.sum(diff**2)

    def jac(x):
        # Gradient of sum(diff^2) = 2 * sum(diff * dRec/dx)
        # This is computationally expensive to implement analytically for speed, 
        # so we might rely on numerical gradients or a simplified ALS step inside minimize?
        # Actually, for 1760 variables, numerical gradient is slow.
        # Let's stick to ALS but with better initialization and more restarts, 
        # OR use scipy's least_squares which handles Jacobian better?
        # No, let's try to improve the ALS convergence.
        # The previous ALS failed likely due to getting stuck in local minima.
        # We can use "Polynomial System Solving" via `np.polynomial`? No.
        
        # Let's try a hybrid: Use ALS to get close, then polish with minimize.
        # Or just increase restarts and add noise to escape local minima.
        
        # For the objective function in minimize, we need it to be fast.
        # Let's use the ALS step as a "line search" or just run ALS and check.
        
        # Actually, let's just make the ALS more robust.
        # 1. More restarts.
        # 2. Random initialization with different scales.
        # 3. If stuck, add noise.
        
        pass # Placeholder, we will implement the search loop below

    deadline = time.time() + 140
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    found_rank = None

    # Try rank 44
    R = 44
    
    max_restarts = 50
    max_iter = 100
    tol = 1e-6

    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Diverse initialization
        scale = 1.0 / np.sqrt(R)
        U = np.random.randn(nm, R) * scale
        V = np.random.randn(mp, R) * scale
        W = np.random.randn(pn, R) * scale
        
        current_err = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update W
            P_UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
            A_W = P_UV.reshape(nm * mp, R)
            W_new = np.zeros((pn, R), dtype=np.float64)
            for w in range(pn):
                b = T[:, :, w].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                W_new[w] = sol
            W = W_new
            
            # Update V
            P_UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]
            A_V = P_UW.reshape(nm * pn, R)
            V_new = np.zeros((mp, R), dtype=np.float64)
            for v in range(mp):
                b = T[:, v, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                V_new[v] = sol
            V = V_new
            
            # Update U
            P_VW = V[:, np.newaxis, :] * W[np.newaxis, :, :]
            A_U = P_VW.reshape(mp * pn, R)
            U_new = np.zeros((nm, R), dtype=np.float64)
            for u in range(nm):
                b = T[u, :, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                U_new[u] = sol
            U = U_new
            
            err = error_fn(U, V, W)
            
            if err < current_err:
                current_err = err
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
            else:
                # If error increased, we might be oscillating or stuck.
                # Add small noise to escape?
                if it > 10 and err > current_err * 1.01:
                    U += np.random.randn(nm, R) * 0.01
                    V += np.random.randn(mp, R) * 0.01
                    W += np.random.randn(pn, R) * 0.01
            
            if err < tol:
                break
        
        if best_err < tol:
            found_rank = R
            break

    if found_rank is not None and best_U is not None:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition of rank n*m*p = 48
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
