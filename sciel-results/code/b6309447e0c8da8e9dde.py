import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Indices:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Precompute indices for fast evaluation
    # Flatten the target tensor
    # Target is all 1s of shape (n, m, p)
    
    # We will try ranks from lower bound up to schoolbook
    # Lower bound is max(n*m, m*p, p*n) = 12
    # Schoolbook is 24
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Helper to evaluate error
    def error_and_grad(params, r):
        # params shape: (r * (n*m + m*p + p*n),)
        # Reshape to U, V, W
        nm = n * m
        mp = m * p
        pn = p * n
        total_vars = r * (nm + mp + pn)
        
        U = params[:r*nm].reshape(r, nm)
        V = params[r*nm:r*nm+r*mp].reshape(r, mp)
        W = params[r*nm+r*mp:].reshape(r, pn)
        
        # Compute the resulting tensor
        # result[i, j, k] = sum_c U[c, i*m+j] * V[c, j*p+k] * W[c, k*n+i]
        
        # Efficient computation using einsum or loops?
        # r is small (12-24), dims are small.
        # Loop over c might be fast enough if vectorized over i,j,k.
        
        # Let's use einsum
        # U is (r, nm), V is (r, mp), W is (r, pn)
        # We need to map indices back to (i,j), (j,k), (k,i)
        
        # This is tricky with einsum directly because of the shared indices j and k.
        # Let's stick to a simple loop over c, which is fine for small r.
        
        result = np.zeros((n, m, p))
        for c in range(r):
            u_vec = U[c] # shape (nm,) -> reshape to (n, m)
            v_vec = V[c] # shape (mp,) -> reshape to (m, p)
            w_vec = W[c] # shape (pn,) -> reshape to (p, n)
            
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            
            # For each i, j, k:
            # term = u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
            # This is a tensor product of vectors, but with index matching.
            # We can compute this as:
            # Temp[i, j, k] = u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
            
            # Vectorized way:
            # u_mat[i,j] is independent of k.
            # v_mat[j,k] is independent of i.
            # w_mat[k,i] is independent of j.
            
            # Let's just use a loop for i, j, k since n,m,p are tiny.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        result[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                        
        error = result - 1.0
        flat_error = error.flatten()
        
        # Gradient calculation is complex. Let's use numerical gradient or just minimize without gradient (Nelder-Mead).
        # Nelder-Mead is slower but doesn't need gradient.
        # However, for 100+ variables, Nelder-Mead might be too slow.
        # Let's use L-BFGS-B with numerical gradient? Or just return value and let scipy handle it?
        # scipy.optimize.minimize can handle non-gradient methods.
        
        return np.sum(flat_error ** 2)

    # We will try ranks in increasing order
    for r in range(12, 25):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        num_restarts = 5 if r < 16 else 3
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize parameters
            # Random initialization
            np.random.seed(42 + r * 10 + restart) # Reproducible
            params = np.random.randn(r * (n*m + m*p + p*n))
            
            # Optimization
            # Use Nelder-Mead for robustness, or L-BFGS-B.
            # L-BFGS-B with numerical gradient is often better for smooth problems.
            # The function is smooth (polynomial).
            
            try:
                res = minimize(error_and_grad, params, args=(r,), method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.success and res.fun < 1e-6:
                # Check if we can snap to half-integers
                # Reshape res.x
                nm = n * m
                mp = m * p
                pn = p * n
                U_opt = res.x[:r*nm].reshape(r, nm)
                V_opt = res.x[r*nm:r*nm+r*mp].reshape(r, mp)
                W_opt = res.x[r*nm+r*mp:].reshape(r, pn)
                
                # Snap to nearest 0.5
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify exactness
                # Reconstruct tensor with snapped values
                result = np.zeros((n, m, p))
                for c in range(r):
                    u_mat = U_snap[c].reshape(n, m)
                    v_mat = V_snap[c].reshape(m, p)
                    w_mat = W_snap[c].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                result[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                
                if np.allclose(result, 1.0, atol=1e-9):
                    # Found a valid decomposition
                    # Convert to list of lists as required
                    U_list = [[0]*r for _ in range(n*m)]
                    V_list = [[0]*r for _ in range(m*p)]
                    W_list = [[0]*r for _ in range(p*n)]
                    
                    for c in range(r):
                        for idx in range(n*m):
                            U_list[idx][c] = U_snap[c, idx]
                        for idx in range(m*p):
                            V_list[idx][c] = V_snap[c, idx]
                        for idx in range(p*n):
                            W_list[idx][c] = W_snap[c, idx]
                            
                    return {"U": U_list, "V": V_list, "W": W_list}
            
    # If no better rank found, return schoolbook
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
