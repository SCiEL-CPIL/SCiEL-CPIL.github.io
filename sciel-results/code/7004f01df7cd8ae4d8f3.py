import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor Y
    # Shape: (n*m, m*p, p*n)
    Y = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                Y[a, b, c] = 1.0

    deadline = time.time() + 130  # Safety margin for 150s limit

    # Try ranks from 45 upwards
    # Rank 45 is the theoretical minimum, but hard to find.
    # Rank 50, 55 are more likely to be found if they exist.
    # We will try a few random initializations for each.
    
    ranks_to_try = [45, 50, 55]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        max_restarts = 10 if rank > 45 else 5
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + restart)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            def objective(params):
                U_flat = params[:n * m * rank].reshape(n * m, rank)
                V_flat = params[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                W_flat = params[n * m * rank + m * p * rank :].reshape(p * n, rank)
                
                T = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
                diff = T - Y
                return np.sum(diff * diff)
            
            def gradient(params):
                U_flat = params[:n * m * rank].reshape(n * m, rank)
                V_flat = params[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                W_flat = params[n * m * rank + m * p * rank :].reshape(p * n, rank)
                
                T = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
                diff = T - Y
                
                gU = np.einsum('abc,br,cr->ar', diff, V_flat, W_flat)
                gV = np.einsum('abc,ar,cr->br', diff, U_flat, W_flat)
                gW = np.einsum('abc,ar,br->cr', diff, U_flat, V_flat)
                
                return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, 
                               options={'maxiter': 100, 'maxfun': 100, 'gtol': 1e-6})
                
                if res.fun < 1e-5:
                    U_opt = res.x[:n * m * rank].reshape(n * m, rank)
                    V_opt = res.x[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                    W_opt = res.x[n * m * rank + m * p * rank :].reshape(p * n, rank)
                    
                    # Round to nearest half-integer
                    U_int = np.round(U_opt * 2) / 2
                    V_int = np.round(V_opt * 2) / 2
                    W_int = np.round(W_opt * 2) / 2
                    
                    # Verify
                    T_check = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                    if np.max(np.abs(T_check - Y)) < 1e-9:
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        
            except Exception as e:
                pass

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
