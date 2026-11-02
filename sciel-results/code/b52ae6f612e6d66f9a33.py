import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Build the matrix multiplication tensor T
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    # i in [0,n), j in [0,m), k in [0,p)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank 30)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W, R):
        """Check if U, V, W form a valid decomposition with half-integer entries."""
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((Rec - T)**2)
        if err > 1e-8:
            return False, err
        # Check half-integer entries
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        if np.allclose(U2, np.round(U2), atol=1e-6) and \
           np.allclose(V2, np.round(V2), atol=1e-6) and \
           np.allclose(W2, np.round(W2), atol=1e-6):
            return True, err
        return False, err
    
    # Try to find rank 29 decomposition
    target_rank = 29
    
    def try_rank(R, max_restarts=10):
        """Try to find a decomposition of rank R."""
        best_err = float('inf')
        best_params = None
        
        np.random.seed(42)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with schoolbook-like structure
            # Use random perturbations of the schoolbook
            params = np.random.randn(R * (n*m + m*p + p*n)) * 0.5
            
            try:
                from scipy.optimize import minimize
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R
                
                def objective(p):
                    U_flat = p[:size_U]
                    V_flat = p[size_U:size_U+size_V]
                    W_flat = p[size_U+size_V:]
                    U = U_flat.reshape(n*m, R)
                    V = V_flat.reshape(m*p, R)
                    W = W_flat.reshape(p*n, R)
                    Rec = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.sum((Rec - T)**2)
                    return err
                
                res = minimize(objective, params, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
            except Exception as e:
                pass
        
        if best_params is not None and best_err < 1e-8:
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
        
        return None
    
    # Try rank 29
    result = try_rank(29, max_restarts=5)
    if result is not None:
        return result
    
    # If rank 29 fails, return schoolbook rank 30
    return {"U": U_sb, "V": V_sb, "W": W_sb}

