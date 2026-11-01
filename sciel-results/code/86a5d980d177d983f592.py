import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Precompute indices for the tensor
    # Tensor T has shape (n, m, m, p, n, p)
    # T[i, j, jp, k, ip, kp] = 1 if (i==ip, j==jp, k==kp) else 0
    # We want to find U (n*m, r), V (m*p, r), W (p*n, r) such that:
    # sum_c U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c] = T[i,j,jp,k,ip,kp]
    
    # Vectorize the tensor for faster residual computation
    # T_flat[i*m + j, jp*m + k, kp*p + ip] -- wait, let's be careful with indices
    # U index: i*m + j
    # V index: jp*p + k
    # W index: kp*n + ip
    
    # Let's build a list of tuples (u_idx, v_idx, w_idx, target)
    # This is sparse. Only 18 non-zero entries.
    # But the residual involves summing over c for ALL entries.
    # So we need to compute the full tensor product.
    
    # Efficient residual:
    # Let A = U (nm x r), B = V (mp x r), C = W (pn x r)
    # We want to check if A ⊗ B ⊗ C matches T.
    # T is a 6D tensor.
    # We can reshape U, V, W to 3D? No, they are 2D.
    # But we can compute the tensor product efficiently using einsum or matmul tricks.
    
    # T[i, j, jp, k, ip, kp] = sum_c U[i*m+j, c] V[jp*p+k, c] W[kp*n+ip, c]
    
    # Let's reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    # Then T = np.einsum('ijc,jkc,kic->ijkkic', ...) -- indices are getting messy.
    # U: i, j, c
    # V: jp, k, c
    # W: kp, ip, c  (Note: W is indexed by kp*n + ip, so first dim is kp, second is ip)
    
    # So:
    # U_reshaped: (n, m, r)
    # V_reshaped: (m, p, r)
    # W_reshaped: (p, n, r)
    
    # T[i, j, jp, k, ip, kp] = sum_c U[i,j,c] V[jp,k,c] W[kp,ip,c]
    # This looks like a tensor contraction.
    # We can compute this as:
    # Temp1[i, j, jp, k, c] = U[i,j,c] * V[jp,k,c]
    # Temp2[i, j, jp, k, ip, kp] = Temp1[i, j, jp, k, c] * W[kp, ip, c]
    # Sum over c.
    
    # This can be done with einsum:
    # np.einsum('ijc,jkc,kpc->ijkkpc', U_reshaped, V_reshaped, W_reshaped)
    # Wait, W is (p, n, r). Index kp is first dim, ip is second dim.
    # So W[kp, ip, c].
    # Let's check the order in the target tensor T[i, j, jp, k, ip, kp].
    # The last two indices are ip, kp.
    # In W, the indices are kp, ip.
    # So we need to be careful.
    
    # Let's define:
    # U: (n, m, r)
    # V: (m, p, r)
    # W: (p, n, r)
    
    # We want T[i, j, jp, k, ip, kp] = sum_c U[i,j,c] V[jp,k,c] W[kp,ip,c]
    # einsum: 'ijc,jkc,kpc->ijkkpc'
    # Let's trace:
    # i from U
    # j from U
    # jp from V
    # k from V
    # ip from W (second dim of W)
    # kp from W (first dim of W)
    # c is summed.
    # Output indices: i, j, jp, k, ip, kp.
    # This matches T[i, j, jp, k, ip, kp].
    
    # So we can compute the predicted tensor quickly.
    
    # Build the target tensor T
    T = np.zeros((n, m, m, p, n, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, i, k] = 1.0
    
    T_flat = T.flatten()
    
    # Define residual using vectorized computation
    def residual(flat_params):
        U_r = flat_params[:n*m].reshape(n, m) # Wait, rank is variable.
        # flat_params contains U, V, W for a specific rank r.
        # We need to pass r or infer it.
        # Let's define a helper that takes U, V, W matrices directly.
        pass

    # Better approach:
    # The optimization problem is non-convex.
    # We can use ALS (Alternating Least Squares) or similar.
    # Or use scipy.optimize.minimize with a good initial guess.
    
    # Let's try a structured search for small ranks.
    # Known lower bound for 2x3 * 3x3 is 15? Or 16?
    # Strassen-like algorithms might exist.
    # Let's try rank 16 first, then 15, etc.
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try ranks from 16 down to 15
    for rank in [16, 15]:
        if time.time() > deadline - 5:
            break
            
        # Try multiple random restarts
        for attempt in range(50):
            if time.time() > deadline - 2:
                break
                
            np.random.seed(attempt + rank)
            
            # Random initialization with small values
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Define objective function
            def objective(params):
                U_r = params[:n*m*rank].reshape(n*m, rank)
                V_r = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                W_r = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
                
                U_reshaped = U_r.reshape(n, m, rank)
                V_reshaped = V_r.reshape(m, p, rank)
                W_reshaped = W_r.reshape(p, n, rank)
                
                # Compute tensor product
                # 'ijc,jkc,kpc->ijkkpc'
                T_pred = np.einsum('ijc,jkc,kpc->ijkkpc', U_reshaped, V_reshaped, W_reshaped, optimize=True)
                
                diff = T_pred - T
                return np.sum(diff * diff)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                from scipy.optimize import minimize
                result = minimize(objective, x0, method='BFGS', options={'maxiter': 100, 'gtol': 1e-6})
                
                if result.fun < 1e-6:
                    U_sol = result.x[:n*m*rank].reshape(n*m, rank)
                    V_sol = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_sol = result.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                    
                    # Round to half-integers
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    
                    # Verify
                    if verify(U_int, V_int, W_int, n, m, p):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        
                    # If rounding failed, try rounding to integers
                    U_int = np.round(U_sol)
                    V_int = np.round(V_sol)
                    W_int = np.round(W_sol)
                    if verify(U_int, V_int, W_int, n, m, p):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        
            except Exception:
                continue
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

def verify(U, V, W, n, m, p):
    # U: (nm, r), V: (mp, r), W: (pn, r)
    r = U.shape[1]
    # Check if U, V, W are half-integers
    if not np.allclose(U * 2, np.round(U * 2)): return False
    if not np.allclose(V * 2, np.round(V * 2)): return False
    if not np.allclose(W * 2, np.round(W * 2)): return False
    
    # Use vectorized verification for speed
    U_reshaped = np.array(U).reshape(n, m, r)
    V_reshaped = np.array(V).reshape(m, p, r)
    W_reshaped = np.array(W).reshape(p, n, r)
    
    T_pred = np.einsum('ijc,jkc,kpc->ijkkpc', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    
    # Check against target
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not np.isclose(T_pred[i, j, j, k, i, k], 1.0, atol=1e-9):
                    return False
                # Check zeros
                for jp in range(m):
                    for kp in range(p):
                        for ip in range(n):
                            if (i, j, k) != (ip, jp, kp):
                                if not np.isclose(T_pred[i, j, jp, k, ip, kp], 0.0, atol=1e-9):
                                    return False
    return True
