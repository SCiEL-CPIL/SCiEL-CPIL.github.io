import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe
    
    n, m, p = DIMENSIONS
    # Tensor dimensions:
    # U: (n*m) x R
    # V: (m*p) x R
    # W: (p*n) x R
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We want U @ V^T @ W^T to approximate T?
    # No, the decomposition is T = sum_r U_r \otimes V_r \otimes W_r
    # This means T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    
    # Create the target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Try to find a decomposition of rank R
    # We will try a few ranks, starting from a known good guess (33)
    # If 33 fails, try 36, 40, etc.
    
    ranks_to_try = [33, 36, 40, 45, 54]
    
    best_result = None
    best_err = float('inf')
    
    for R in ranks_to_try:
        if time.time() > deadline - 20.0:
            break
            
        # Initialize U, V, W
        # Use random initialization
        np.random.seed(42)
        U = np.random.randn(dim_u, R)
        V = np.random.randn(dim_v, R)
        W = np.random.randn(dim_w, R)
        
        # ALS iterations
        max_iters = 50
        tol = 1e-10
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            # Update U: T_{a,:,:} = sum_r U[a,r] (V[:,r] \otimes W[:,r])^T
            # T_mat (dim_u, dim_v*dim_w)
            T_u = T.reshape(dim_u, dim_v * dim_w)
            VW = (V * W).reshape(dim_v * dim_w, R) # V[:,r]*W[:,r]
            # U = T_u @ VW @ inv(VW^T @ VW)
            try:
                U_new = T_u @ VW @ np.linalg.inv(VW.T @ VW)
                U = U_new
            except np.linalg.LinAlgError:
                pass
                
            # Update V
            T_v = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
            UW = (U * W).reshape(dim_u * dim_w, R)
            try:
                V_new = T_v @ UW @ np.linalg.inv(UW.T @ UW)
                V = V_new
            except np.linalg.LinAlgError:
                pass
                
            # Update W
            T_w = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
            UV = (U * V).reshape(dim_u * dim_v, R)
            try:
                W_new = T_w @ UV @ np.linalg.inv(UV.T @ UV)
                W = W_new
            except np.linalg.LinAlgError:
                pass
                
            # Check convergence
            # Error = ||T - sum U_r V_r W_r||
            # We can estimate error by sampling or computing full norm
            # Full norm is expensive but dims are small (6, 27, 18)
            # T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            # err = np.linalg.norm(T - T_approx)
            
            # Optimization: compute error only every 10 iterations or at the end
            if it % 10 == 0 or it == max_iters - 1:
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T - T_approx)
                if err < tol:
                    break
                    
        # Final error check
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - T_approx)
        
        if err < best_err:
            best_err = err
            best_result = (U, V, W, R)
            
        if best_err < 1e-8:
            break
            
    if best_result is None:
        # Fallback to schoolbook
        R = n * m * p
        U = np.zeros((dim_u, R))
        V = np.zeros((dim_v, R))
        W = np.zeros((dim_w, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W, R = best_result
        # Convert to list of lists with integer/half-integer if possible
        # The problem allows integer/half-integer. 
        # Numerical results might be floating point.
        # We should round to nearest half-integer if close.
        def clean_matrix(M):
            M_clean = np.zeros_like(M)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    val = M[i, j]
                    # Try rounding to nearest 0.5
                    half_int = np.round(val * 2) / 2
                    if abs(val - half_int) < 1e-6:
                        M_clean[i, j] = half_int
                    else:
                        M_clean[i, j] = val
            return M_clean
            
        U_clean = clean_matrix(U)
        V_clean = clean_matrix(V)
        W_clean = clean_matrix(W)
        
        return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}

