import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find rank 29
    target_rank = 29
    if time.time() > 140: # Safety
        return {"U": U_sb, "V": V_sb, "W": W_sb}
        
    # Build Tensor T
    # T[a, b, c] = 1 if i_a==i_c, j_a==j_b, k_b==k_c
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Function to check if a decomposition is valid (close to integer/half-integer and exact)
    def check_decomposition(U, V, W, R):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Compute reconstruction
        # Rec[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Efficient way:
        # For each r, add outer product
        # Or use einsum
        # Rec = np.einsum('ar,br,cr->abc', U, V, W)
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((Rec - T)**2)
        if err > 1e-8:
            return False, err
        # Check if entries are close to half-integers
        # Multiply by 2, round, check
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        if np.allclose(U2, np.round(U2), atol=1e-6) and \
           np.allclose(V2, np.round(V2), atol=1e-6) and \
           np.allclose(W2, np.round(W2), atol=1e-6):
            # Convert to list of lists of float (half-integers)
            U_out = [[float(x/2) for x in row] for row in U2]
            V_out = [[float(x/2) for x in row] for row in V2]
            W_out = [[float(x/2) for x in row] for row in W2]
            return True, err
        return False, err

    # Optimization
    # Variables: U, V, W flattened
    # Total vars: R * (nm + mp + pn)
    # For R=29: 29 * (6 + 15 + 10) = 29 * 31 = 899
    
    def objective(params):
        R = target_rank
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((Rec - T)**2)
        return err

    best_err = float('inf')
    best_params = None
    
    # Try a few random restarts
    for restart in range(10):
        if time.time() > 145:
            break
        # Random init
        np.random.seed(42 + restart)
        params = np.random.randn(target_rank * (n*m + m*p + p*n))
        
        # Use scipy.optimize
        try:
            from scipy.optimize import minimize
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
        except Exception as e:
            pass

    if best_params is not None and best_err < 1e-6:
        R = target_rank
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        U_flat = best_params[:size_U]
        V_flat = best_params[size_U:size_U+size_V]
        W_flat = best_params[size_U+size_V:]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        valid, err = check_decomposition(U, V, W, R)
        if valid:
            return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
