import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 115.0  # Leave 5s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    # T[i, j, j2, k, i2, k2] = 1 if j==j2 and i==i2 and k==k2 else 0
    # Shape: (n, m, m, p, n, p)
    T = np.zeros((n, m, m, p, n, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, i, k] = 1.0
                
    T_flat = T.flatten()
    
    # Precompute schoolbook basis vectors for U, V, W
    # There are n*m*p basis vectors.
    # For a basis vector indexed by (i, j, k):
    # U[i*m + j] = 1, V[j*p + k] = 1, W[k*n + i] = 1
    num_basis = n * m * p
    U_basis = np.zeros((num_basis, n * m))
    V_basis = np.zeros((num_basis, m * p))
    W_basis = np.zeros((num_basis, p * n))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_basis[idx, i * m + j] = 1.0
                V_basis[idx, j * p + k] = 1.0
                W_basis[idx, k * n + i] = 1.0
                idx += 1

    def residual_fn(params, R):
        # params: flattened U, V, W
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape((n * m, R)).T # (R, n*m)
        V = V_flat.reshape((m * p, R)).T # (R, m*p)
        W = W_flat.reshape((p * n, R)).T # (R, p*n)
        
        U_reshaped = U.reshape((R, n, m))
        V_reshaped = V.reshape((R, m, p))
        W_reshaped = W.reshape((R, p, n))
        
        recon = np.zeros_like(T)
        for r in range(R):
            u = U_reshaped[r] # (n, m)
            v = V_reshaped[r] # (m, p)
            w = W_reshaped[r] # (p, n)
            
            uv = np.einsum('ij,jk->ijjk', u, v)
            recon += np.einsum('ijjk,ab->ijkab', uv, w)
            
        return (recon.flatten() - T_flat)
        
    best_solution = None
    best_rank = n * m * p
    
    # Try ranks from 45 to 83
    # Lower bound is roughly max(n*m, m*p, n*p) = 28, but tensor rank is usually higher.
    # For 3x4x7, let's start from 45 to save time, but go down to 40 if fast.
    # We will iterate from low to high.
    
    # To improve search, we can use a mix of random noise and schoolbook-based init
    for R in range(45, min(84, n*m*p)):
        if time.time() > deadline:
            break
            
        # Multiple restarts
        # If R is close to 84, fewer restarts might be needed if it's easy, but let's keep it consistent.
        # If R is low, we need more restarts.
        num_restarts = 10 if R < 60 else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            total_size = size_U + size_V + size_W
            
            if restart == 0:
                # Initialization: Random linear combination of schoolbook basis
                # Pick R random basis vectors and combine them?
                # Or generate random coefficients for R basis vectors?
                # Let's generate R random vectors in the span of the basis.
                # This is equivalent to:
                # U_init = U_basis @ C_U, where C_U is (num_basis, R)
                # But we need U_init to be (n*m, R).
                # U_basis is (num_basis, n*m).
                # So U_init = U_basis.T @ C_U ? No.
                # U_basis[i, row] = 1 if basis i has 1 at row.
                # We want U_init[row, r] = sum_{l} c_{l,r} * U_basis[l, row]
                # So U_init = U_basis.T @ C, where C is (num_basis, R).
                
                C_U = np.random.randn(num_basis, R) * 0.5
                C_V = np.random.randn(num_basis, R) * 0.5
                C_W = np.random.randn(num_basis, R) * 0.5
                
                U_init = U_basis.T @ C_U # (n*m, R)
                V_init = V_basis.T @ C_V # (m*p, R)
                W_init = W_basis.T @ C_W # (p*n, R)
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            else:
                # Random initialization
                x0 = np.random.randn(total_size) * 0.5
            
            try:
                res = least_squares(residual_fn, x0, args=(R,), method='lm', max_nfev=500)
            except Exception:
                continue
                
            if res.cost < 1e-10:
                # Check if we can snap to half-integers
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                U_snap = np.round(U_flat * 2) / 2
                V_snap = np.round(V_flat * 2) / 2
                W_snap = np.round(W_flat * 2) / 2
                
                # Verify snapped solution
                x_snapped = np.concatenate([U_snap, V_snap, W_snap])
                res_snapped = residual_fn(x_snapped, R)
                
                if np.max(np.abs(res_snapped)) < 1e-9:
                    # Valid!
                    best_rank = R
                    best_solution = {
                        "U": U_snap.reshape((n*m, R)).tolist(),
                        "V": V_snap.reshape((m*p, R)).tolist(),
                        "W": W_snap.reshape((p*n, R)).tolist()
                    }
                    break
        if best_solution:
            break

    if best_solution:
        return best_solution
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
