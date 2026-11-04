import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build the target tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if this term contributes to C[i, k]
    # Actually, the tensor T represents the multiplication operation.
    # The term A[i, j] * B[j, k] contributes to C[i, k].
    # So T[i, j, k] = 1.
    T = np.ones((n, m, p))
    
    # Function to flatten indices for U, V, W
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    def reconstruct(params, R):
        # params is a flat vector of length R * (n*m + m*p + p*n)
        # We split it into U, V, W
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U_flat = params[0:R*size_U]
        V_flat = params[R*size_U:R*size_U + R*size_V]
        W_flat = params[R*size_U + R*size_V:R*size_U + R*size_V + R*size_W]
        
        U = U_flat.reshape(R, size_U)
        V = V_flat.reshape(R, size_V)
        W = W_flat.reshape(R, size_W)
        
        # Compute the reconstructed tensor
        # T_recon[i, j, k] = sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
        T_recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Sum over r
                    T_recon[i, j, k] = np.sum(U[:, u_idx] * V[:, v_idx] * W[:, w_idx])
        return T_recon

    def loss(params, R):
        T_recon = reconstruct(params, R)
        error = np.sum((T - T_recon) ** 2)
        return error

    def snap_and_verify(params, R):
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U_flat = params[0:R*size_U]
        V_flat = params[R*size_U:R*size_U + R*size_V]
        W_flat = params[R*size_U + R*size_V:R*size_U + R*size_V + R*size_W]
        
        # Snap to half-integers
        U_snap = np.round(U_flat * 2) / 2
        V_snap = np.round(V_flat * 2) / 2
        W_snap = np.round(W_flat * 2) / 2
        
        # Check exactness
        T_recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T_recon[i, j, k] = np.sum(U_snap[u_idx] * V_snap[v_idx] * W_snap[w_idx])
                    
        if np.allclose(T_recon, T, atol=1e-9):
            # Format output
            U_out = [[float(U_snap[i*size_U + r]) for r in range(R)] for i in range(size_U)]
            V_out = [[float(V_snap[i*size_V + r]) for r in range(R)] for i in range(size_V)]
            W_out = [[float(W_snap[i*size_W + r]) for r in range(R)] for i in range(size_W)]
            return {"U": U_out, "V": V_out, "W": W_out}
        return None

    # Try ranks from schoolbook_rank - 1 down to a minimum reasonable bound
    # The minimum rank cannot be less than the number of non-zero entries in any slice?
    # A lower bound is max(n*m, m*p, p*n) ? No.
    # For 2x3x4, lower bound is at least 12?
    # Let's try R from 23 down to 18.
    
    best_result = None
    
    for R in range(schoolbook_rank - 1, 17, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Random initialization
            np.random.seed(attempt * 100 + R)
            # Small random values to avoid huge gradients
            init = np.random.randn(R * (n*m + m*p + p*n)) * 0.5
            
            # Use BFGS or Nelder-Mead
            # BFGS is faster but needs gradients. Let's use Nelder-Mead for simplicity, 
            # or calculate gradients.
            # Given small dimension, Nelder-Mead might be okay, but BFGS is better.
            # Let's use a simple gradient step with numerical gradient or analytic.
            
            # Analytic gradient:
            # L = sum (T - sum_r U V W)^2
            # dL/dU[r, u] = -2 * (T - T_recon) * V[r, v] * W[r, w]
            
            def gradient(params, R):
                T_recon = reconstruct(params, R)
                diff = T - T_recon
                size_U = n * m
                size_V = m * p
                size_W = p * n
                
                U_flat = params[0:R*size_U]
                V_flat = params[R*size_U:R*size_U + R*size_V]
                W_flat = params[R*size_U + R*size_V:R*size_U + R*size_V + R*size_W]
                
                U = U_flat.reshape(R, size_U)
                V = V_flat.reshape(R, size_V)
                W = W_flat.reshape(R, size_W)
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            d = diff[i, j, k]
                            # grad_U[r, u_idx] += -2 * d * V[r, v_idx] * W[r, w_idx]
                            grad_U[:, u_idx] -= 2 * d * (V[:, v_idx] * W[:, w_idx])
                            grad_V[:, v_idx] -= 2 * d * (U[:, u_idx] * W[:, w_idx])
                            grad_W[:, w_idx] -= 2 * d * (U[:, u_idx] * V[:, v_idx])
                
                grad_flat = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                return grad_flat

            # Optimization
            try:
                res = minimize(loss, init, args=(R,), method='BFGS', jac=gradient, options={'maxiter': 500, 'disp': False})
                if res.success or loss(res.x, R) < 1e-6:
                    res_params = res.x
                    result = snap_and_verify(res_params, R)
                    if result:
                        best_result = result
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            return best_result

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
