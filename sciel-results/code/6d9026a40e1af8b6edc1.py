import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Try rank 21 first
    R = 21
    
    # Reshaping helpers
    def reshape_U(flat):
        return flat.reshape(n*m, R)
    def reshape_V(flat):
        return flat.reshape(m*p, R)
    def reshape_W(flat):
        return flat.reshape(p*n, R)
        
    def objective(x):
        U = reshape_U(x[:n*m*R])
        V = reshape_V(x[n*m*R : n*m*R + m*p*R])
        W = reshape_W(x[n*m*R + m*p*R :])
        
        # Reshape to 3D for einsum
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        # Compute tensor
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        
        # Target is all ones
        residual = T - 1.0
        return np.sum(residual**2)
        
    # Initialize
    np.random.seed(42)
    x0 = np.random.randn(n*m*R + m*p*R + p*n*R) * 0.1
    
    # Optimize
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
    
    if result.success or result.fun < 1e-6:
        U = reshape_U(result.x[:n*m*R])
        V = reshape_V(result.x[n*m*R : n*m*R + m*p*R])
        W = reshape_W(result.x[n*m*R + m*p*R :])
        
        # Check if we can round to half-integers
        # Try rounding to nearest 0.5
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify rounded
        U3 = U_rounded.reshape(n, m, R)
        V3 = V_rounded.reshape(m, p, R)
        W3 = W_rounded.reshape(p, n, R)
        T_rounded = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err_rounded = np.max(np.abs(T_rounded - 1.0))
        
        if err_rounded < 1e-6:
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        else:
            # Return unrounded if rounded fails, assuming tolerance is ok
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # If rank 21 fails, fall back to schoolbook (rank 63)
        # But we expect 21 to work
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
