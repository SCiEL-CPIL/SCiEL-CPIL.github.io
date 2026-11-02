import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Target rank
    target_rank = 29
    
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

    def check_decomposition(U, V, W, R):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Rec[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((Rec - T)**2)
        if err > 1e-6:
            return False, err
        # Check if entries are close to half-integers
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        if np.allclose(U2, np.round(U2), atol=1e-4) and \
           np.allclose(V2, np.round(V2), atol=1e-4) and \
           np.allclose(W2, np.round(W2), atol=1e-4):
            # Convert to list of lists of float (half-integers)
            U_out = [[float(x/2) for x in row] for row in U2]
            V_out = [[float(x/2) for x in row] for row in V2]
            W_out = [[float(x/2) for x in row] for row in W2]
            return True, err
        return False, err

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
        
        # Regularization for half-integers
        reg = 0.0
        for X in [U, V, W]:
            X2 = X * 2
            reg += np.sum((X2 - np.round(X2))**2)
        
        return err + 0.1 * reg

    best_err = float('inf')
    best_params = None
    
    start_time = time.time()
    deadline = 140.0
    
    # Try a few random restarts
    for restart in range(5):
        if time.time() > deadline:
            break
        
        np.random.seed(42 + restart)
        # Initialize near schoolbook? No, schoolbook is rank 30.
        # Random init
        params = np.random.randn(target_rank * (n*m + m*p + p*n)) * 0.5
        
        try:
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'maxls': 20})
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
        except Exception as e:
            pass

    if best_params is not None and best_err < 1e-4:
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
