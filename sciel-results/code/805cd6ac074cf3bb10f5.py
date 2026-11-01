import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 3, 4
    rank = 12
    
    # Tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    # Reshape T to (n*m, p) ? No, the tensor is in (i,j) x (j,k) x (k,i)
    # We want U (nm x r), V (mp x r), W (pn x r)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Flatten T to a vector of length n*m*p
    T_flat = T.flatten()
    
    def loss(params):
        # params: flattened U, V, W
        # U: 6x12, V: 12x12, W: 8x12
        # Total size: 6*12 + 12*12 + 8*12 = 72 + 144 + 96 = 312
        
        U = params[:72].reshape(6, 12)
        V = params[72:216].reshape(12, 12)
        W = params[216:312].reshape(8, 12)
        
        # Compute tensor
        # T_comp[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        T_comp = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    T_comp[i, j, k] = s
        
        # Loss: sum of squares of difference
        diff = T_comp.flatten() - T_flat
        return np.sum(diff**2)
        
    # Random initialization
    np.random.seed(42)
    # Try to initialize with integers
    params = np.random.randint(-2, 3, size=312).astype(float)
    
    # Use Nelder-Mead or Powell
    res = minimize(loss, params, method='Powell', options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-8})
    
    # Check if solution is good
    if res.fun < 1e-4:
        # Round to nearest half-integer or integer
        # The solution might be integer
        U = res.x[:72].reshape(6, 12)
        V = res.x[72:216].reshape(12, 12)
        W = res.x[216:312].reshape(8, 12)
        
        # Round
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify
        T_comp = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    T_comp[i, j, k] = s
        
        if np.allclose(T_comp, T, atol=1e-6):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
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
