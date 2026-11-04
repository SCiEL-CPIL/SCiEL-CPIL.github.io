import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor M[a, b, c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    M = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a, b, c] = 1.0

    def residual(params, rank):
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        
        U = params[:size_u].reshape(n*m, rank)
        V = params[size_u:size_u+size_v].reshape(m*p, rank)
        W = params[size_u+size_v:].reshape(p*n, rank)
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        return (T - M).flatten()

    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin for safety

    # Try ranks from 23 to 35
    # Schoolbook is 36.
    for rank in range(23, 36):
        if time.time() > deadline:
            break
            
        # More restarts for lower ranks as they are harder to find
        # But limited by time
        n_restarts = 100 if rank < 26 else 50
        
        for attempt in range(n_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + rank * 100 + attempt)
            # Initialize with small random values
            # Scaling: The values in the solution are likely small integers.
            # Let's try initializing with values in [-1, 1]
            params = np.random.uniform(-1, 1, size=n*m*rank + m*p*rank + p*n*rank)
            
            try:
                # Use LM for speed
                result = least_squares(residual, params, args=(rank,), method='lm', max_nfev=5000)
                
                # Check if solution is good
                if np.max(np.abs(result.fun)) < 1e-8:
                    # Extract U, V, W
                    size_u = n * m * rank
                    size_v = m * p * rank
                    size_w = p * n * rank
                    
                    U = result.x[:size_u].reshape(n*m, rank)
                    V = result.x[size_u:size_u+size_v].reshape(m*p, rank)
                    W = result.x[size_u+size_v:].reshape(p*n, rank)
                    
                    # Try to round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify the rounded solution
                    T_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                    if np.max(np.abs(T_rounded - M)) < 1e-9:
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    
            except Exception:
                pass

    # Fallback: Schoolbook method (Rank 36)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
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
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
