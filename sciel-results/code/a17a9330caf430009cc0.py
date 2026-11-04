import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 45
    
    # Build schoolbook decomposition
    def build_schoolbook():
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-6):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if not np.isclose(val, 1.0, atol=tol):
                        return False
        return True
    
    # Try to find a lower rank decomposition
    # Strategy: Start from schoolbook, randomly perturb, then optimize
    # Use a smooth objective: sum of squared errors
    
    best_U, best_V, best_W = build_schoolbook()
    
    # Try ranks from 44 down to 40
    for R in range(44, 39, -1):
        if time.time() > deadline - 30:
            break
        
        # Multiple restarts
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + R * 100 + restart)
            
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            # Initialize with random values, but bias towards schoolbook structure
            # Use a mixture: some terms are schoolbook-like, some are random
            
            # For simplicity, just use random initialization with small values
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat = x[:n*m*R].reshape(n*m, R)
                V_flat = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_flat = x[n*m*R + m*p*R:].reshape(p*n, R)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                            error += (val - 1.0)**2
                return error
            
            def gradient(x):
                U_flat = x[:n*m*R].reshape(n*m, R)
                V_flat = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_flat = x[n*m*R + m*p*R:].reshape(p*n, R)
                
                grad_U = np.zeros_like(U_flat)
                grad_V = np.zeros_like(V_flat)
                grad_W = np.zeros_like(W_flat)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u = U_flat[i*m+j]
                            v = V_flat[j*p+k]
                            w = W_flat[k*n+i]
                            val = np.dot(u, v * w)
                            err = 2 * (val - 1.0)
                            
                            # d/dU: err * v * w
                            grad_U[i*m+j] += err * v * w
                            
                            # d/dV: err * u * w
                            grad_V[j*p+k] += err * u * w
                            
                            # d/dW: err * u * v
                            grad_W[k*n+i] += err * u * v
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            try:
                res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-8:
                    U_sol = res.x[:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    # Try to round to half-integers
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    if verify(U_rounded, V_rounded, W_rounded, tol=1e-9):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    
                    # If rounding doesn't work, try without rounding (but task requires integer/half-integer)
                    # For now, continue searching
            except Exception as e:
                pass
    
    # Return schoolbook if no better found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
