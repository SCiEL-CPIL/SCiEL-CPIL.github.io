import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    size_u = n * m
    size_v = m * p
    size_w = p * n
    total_size = size_u + size_v + size_w

    def error_and_grad(params, R):
        U = params[:R * size_u].reshape(R, size_u).T  # (size_u, R)
        V = params[R * size_u : R * (size_u + size_v)].reshape(R, size_v).T
        W = params[R * (size_u + size_v) :].reshape(R, size_w).T
        
        # Compute T
        T = np.zeros((n, m, p))
        # Pre-compute components for gradient
        # grad_U[i, j, r] = 2 * (T[i, j, k] - 1) * V[j, k, r] * W[k, i, r]
        # This sum is over k.
        
        # Vectorized T calculation
        # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        # T[i, j, k] = sum_r U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
        # We can compute this via einsum or loops over R
        # Since R is small (< 60), loop over R is fine
        for r in range(R):
            u_r = U3[:, :, r] # (n, m)
            v_r = V3[:, :, r] # (m, p)
            w_r = W3[:, :, r] # (p, n)
            
            # Term for this r: u_r[i, j] * v_r[j, k] * w_r[k, i]
            # This is an outer product-like operation
            # T_r[i, j, k] = u_r[i, j] * v_r[j, k] * w_r[k, i]
            # We can compute this as:
            # temp[i, j, k] = u_r[i, j] * (v_r[j, k] * w_r[k, i])
            # v_r[j, k] * w_r[k, i] is a contraction over k? No, i, j, k are free.
            # Let's just use broadcasting.
            # u_r[i, j, None] * v_r[None, j, k] * w_r[k, None, i]
            # Shapes: (n, m, 1) * (1, m, p) * (p, 1, n) -> (n, m, p)
            term = u_r[:, :, None] * v_r[None, :, :] * w_r.T[None, :, :] # w_r.T is (n, p), so w_r.T[None, :, :] is (1, n, p)? No.
            # w_r is (p, n). w_r.T is (n, p).
            # We need w_r[k, i].
            # So w_r.T[i, k].
            # Shape (n, 1, p)
            w_term = w_r.T[:, None, :] # (n, 1, p)
            
            T += u_r[:, :, None] * v_r[None, :, :] * w_term
            
        diff = T - 1.0
        error = np.sum(diff * diff)
        
        # Gradient
        # dE/dU[i, j, r] = 2 * diff[i, j, k] * V[j, k, r] * W[k, i, r] (sum over k)
        # dE/dV[j, k, r] = 2 * diff[i, j, k] * U[i, j, r] * W[k, i, r] (sum over i)
        # dE/dW[k, i, r] = 2 * diff[i, j, k] * U[i, j, r] * V[j, k, r] (sum over j)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(R):
            u_r = U3[:, :, r] # (n, m)
            v_r = V3[:, :, r] # (m, p)
            w_r = W3[:, :, r] # (p, n)
            
            # For grad_U:
            # We need sum_k diff[i, j, k] * v_r[j, k] * w_r[k, i]
            # diff is (n, m, p)
            # v_r is (m, p) -> (1, m, p)
            # w_r is (p, n) -> (n, 1, p) ?? No, w_r[k, i].
            # Let's reshape diff to (n, m, p).
            # v_r[None, :, :] is (1, m, p)
            # w_r.T[:, None, :] is (n, 1, p)
            # Product: (n, m, p) * (1, m, p) * (n, 1, p) -> (n, m, p)
            # Sum over k (axis 2)
            
            prod_u = diff * v_r[None, :, :] * w_r.T[:, None, :]
            grad_U_r = 2 * np.sum(prod_u, axis=2) # (n, m)
            
            # For grad_V:
            # sum_i diff[i, j, k] * u_r[i, j] * w_r[k, i]
            # u_r[:, :, None] is (n, m, 1)
            # w_r.T[:, None, :] is (n, 1, p)
            # diff is (n, m, p)
            prod_v = diff * u_r[:, :, None] * w_r.T[:, None, :]
            grad_V_r = 2 * np.sum(prod_v, axis=0) # (m, p)
            
            # For grad_W:
            # sum_j diff[i, j, k] * u_r[i, j] * v_r[j, k]
            # u_r[:, :, None] is (n, m, 1)
            # v_r[None, :, :] is (1, m, p)
            prod_w = diff * u_r[:, :, None] * v_r[None, :, :]
            grad_W_r = 2 * np.sum(prod_w, axis=1) # (n, p) -> reshape to (p, n)
            
            grad_U[:, r] = grad_U_r.flatten()
            grad_V[:, r] = grad_V_r.flatten()
            grad_W[:, r] = grad_W_r.T.flatten() # W is stored as (p*n, R), index k*n+i
            # Wait, W3 was reshaped from W (p*n, R) to (p, n, R).
            # So W3[k, i, r] corresponds to W[k*n+i, r].
            # grad_W_r is (n, p) with indices i, k.
            # We need to flatten to match W's layout (k*n+i).
            # So we need grad_W_r.T which is (p, n), then flatten.
            
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return error, grad

    def round_and_check(U, V, W, R):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        T = np.zeros((n, m, p))
        for r in range(R):
            u = U_r[:, r].reshape(n, m)
            v = V_r[:, r].reshape(m, p)
            w = W_r[:, r].reshape(p, n)
            w_term = w.T.reshape(n, 1, p)
            T += u[:, :, None] * v[None, :, :] * w_term
        
        if np.allclose(T, 1.0, atol=1e-9):
            return True, U_r, V_r, W_r
        return False, U_r, V_r, W_r

    def try_rank(R, max_attempts=20):
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            # Scale matters. Start small.
            params = np.random.uniform(-1, 1, R * total_size)
            
            try:
                # Use L-BFGS-B or BFGS with gradient
                # L-BFGS-B is good for large problems, but BFGS is fine here.
                # Let's use L-BFGS-B for robustness.
                res = minimize(error_and_grad, params, args=(R,), method='L-BFGS-B', 
                               jac=True, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.fun < 1e-5:
                U = res.x[:R * size_u].reshape(R, size_u).T
                V = res.x[R * size_u : R * (size_u + size_v)].reshape(R, size_v).T
                W = res.x[R * (size_u + size_v) :].reshape(R, size_w).T
                
                is_valid, U_r, V_r, W_r = round_and_check(U, V, W, R)
                if is_valid:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        return None

    # Try ranks from 45 to 59
    for R in range(45, 60):
        if time.time() > deadline:
            break
        result = try_rank(R)
        if result:
            return result

    # Fallback to schoolbook rank 60
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
