import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
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
                
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    def check_and_finalize(U, V, W, R):
        # Check if entries are close to half-integers
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        # Use a slightly tighter tolerance for rounding
        if np.allclose(U2, np.round(U2), atol=1e-5) and \
           np.allclose(V2, np.round(V2), atol=1e-5) and \
           np.allclose(W2, np.round(W2), atol=1e-5):
            U_out = np.round(U2) / 2.0
            V_out = np.round(V2) / 2.0
            W_out = np.round(W2) / 2.0
            # Verify exactness
            Rec = np.einsum('ar,br,cr->abc', U_out, V_out, W_out)
            if np.allclose(Rec, T, atol=1e-9):
                # Convert to list of lists
                return {
                    "U": U_out.tolist(),
                    "V": V_out.tolist(),
                    "W": W_out.tolist()
                }
        return None

    def objective(params, R):
        size_U = n * m * R
        size_V = m * p * R
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((Rec - T)**2)
        return err

    best_result = None
    
    # Try Rank 28 first, then 29
    target_ranks = [28, 29]
    
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        # Number of restarts depends on time available
        # R=28 is harder, so we give it more time/restarts
        if R == 28:
             max_restarts = 300
             max_iter = 100
        else:
             max_restarts = 500
             max_iter = 100
             
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(restart)
            # Random initialization
            params = np.random.randn(R * (n*m + m*p + p*n)) * 0.5
            
            try:
                from scipy.optimize import minimize
                res = minimize(lambda p: objective(p, R), params, method='L-BFGS-B', 
                               options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-5:
                    # Potential solution found, check if it rounds to half-integers
                    U_flat = res.x[:n*m*R]
                    V_flat = res.x[n*m*R:n*m*R+m*p*R]
                    W_flat = res.x[n*m*R+m*p*R:]
                    
                    U = U_flat.reshape(n*m, R)
                    V = V_flat.reshape(m*p, R)
                    W = W_flat.reshape(p*n, R)
                    
                    result = check_and_finalize(U, V, W, R)
                    if result:
                        best_result = result
                        return best_result
                        
            except Exception:
                pass

    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
