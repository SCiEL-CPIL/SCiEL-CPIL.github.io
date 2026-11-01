import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Build the multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n*m):
        for j in range(m*p):
            for k in range(p*n):
                if (i % m == j // p) and (i // m == k // n) and (j % p == k % n):
                    T[i, j, k] = 1.0
    
    # Schoolbook rank 12
    U_school = np.zeros((4, 12))
    V_school = np.zeros((6, 12))
    W_school = np.zeros((6, 12))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    # Try to find rank 11 using optimization
    # Flatten U, V, W into a single vector
    # U: 4x11 = 44 params, V: 6x11 = 66 params, W: 6x11 = 66 params
    # Total: 176 params
    
    def compute_tensor(params):
        U = params[0:44].reshape(4, 11)
        V = params[44:110].reshape(6, 11)
        W = params[110:176].reshape(6, 11)
        # Compute T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        result = np.zeros((4, 6, 6))
        for i in range(4):
            for j in range(6):
                for k in range(6):
                    result[i, j, k] = np.dot(U[i, :], V[j, :] * W[k, :])
        return result
    
    def objective(params):
        computed = compute_tensor(params)
        return np.sum((computed - T)**2)
    
    # Initialize with schoolbook-like structure
    np.random.seed(123)
    best_params = None
    best_obj = float('inf')
    
    for trial in range(50):
        # Initialize with small random values
        params0 = np.random.randn(176) * 0.5
        result = minimize(objective, params0, method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    
    # Check if we found a valid rank-11 decomposition
    if best_params is not None and best_obj < 1e-8:
        U = best_params[0:44].reshape(4, 11)
        V = best_params[44:110].reshape(6, 11)
        W = best_params[110:176].reshape(6, 11)
        
        # Try to round to integers or half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Check if rounded version still works
        computed = np.zeros((4, 6, 6))
        for i in range(4):
            for j in range(6):
                for k in range(6):
                    computed[i, j, k] = np.dot(U_rounded[i, :], V_rounded[j, :] * W_rounded[k, :])
        
        if np.max(np.abs(computed - T)) < 1e-10:
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
        else:
            # Use unrounded values
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    else:
        # Fall back to schoolbook
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
# EVOLVE-BLOCK-END
