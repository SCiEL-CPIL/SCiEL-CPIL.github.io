import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T
    # T[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # Flattened index for T:
    # The tensor equation is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # We want to find rank r decomposition.
    # Start with r=16.
    
    deadline = time.time() + 100 # 100s limit to be safe
    
    def error_fn(params, r):
        # params: flat array of length r * (n*m + m*p + p*n)
        # Split into U, V, W
        nm = n * m
        mp = m * p
        pn = p * n
        total_vars = r * (nm + mp + pn)
        
        U = params[:r*nm].reshape(r, nm)
        V = params[r*nm:r*nm+r*mp].reshape(r, mp)
        W = params[r*nm+r*mp:].reshape(r, pn)
        
        # Compute product
        # prod[i, j, k] = sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
        # We can vectorize this using einsum or loops.
        # Since r is small (16) and dimensions are small, loops are fine.
        prod = np.zeros((n, m, p))
        for r_idx in range(r):
            u = U[r_idx]
            v = V[r_idx]
            w = W[r_idx]
            for i in range(n):
                for j in range(m):
                    u_val = u[i*m + j]
                    if u_val == 0: continue
                    for k in range(p):
                        v_val = v[j*p + k]
                        if v_val == 0: continue
                        w_val = w[k*n + i]
                        prod[i, j, k] += u_val * v_val * w_val
                        
        # Target is 1 everywhere
        diff = prod - 1.0
        return np.sum(diff**2)

    def try_rank(r):
        if time.time() > deadline:
            return None
            
        best_params = None
        best_err = float('inf')
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Initialize with small random values
            # Scale: 1/sqrt(r) might be good?
            scale = 1.0 / np.sqrt(r)
            params = np.random.randn(r * (n*m + m*p + p*n)) * scale
            
            # Use gradient descent or L-BFGS
            # L-BFGS is faster for this size
            try:
                from scipy.optimize import minimize
                res = minimize(error_fn, params, args=(r,), method='L-BFGS-B', 
                               options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
            except Exception as e:
                pass
            
            # Also try a simple gradient descent step if scipy fails or as backup
            # But L-BFGS should be fine.
            
        if best_params is None:
            return None
            
        if best_err < 1e-6:
            # Try to round to half-integers
            # Rounding to nearest 0.5
            rounded_params = np.round(best_params * 2) / 2.0
            err_rounded = error_fn(rounded_params, r)
            if err_rounded < 1e-9:
                # Valid solution found
                nm = n * m
                mp = m * p
                pn = p * n
                U = rounded_params[:r*nm].reshape(nm, r)
                V = rounded_params[r*nm:r*nm+r*mp].reshape(mp, r)
                W = rounded_params[r*nm+r*mp:].reshape(pn, r)
                # Convert to lists of lists
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Try rank 16
    res16 = try_rank(16)
    if res16:
        return res16

    # Try rank 17
    res17 = try_rank(17)
    if res17:
        return res17

    # Fallback: Schoolbook rank 18
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
