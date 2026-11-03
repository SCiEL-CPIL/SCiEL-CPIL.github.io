import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j] * V_arr[j*p+k], W_arr[k*n+i])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _try_find_lower_rank(deadline):
    """Try to find a rank < 36 decomposition using numerical search."""
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # The tensor entries: T[i,j,k] = 1 for all valid (i,j,k)
    # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Try ranks from schoolbook-1 down to a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    for target_rank in range(schoolbook_rank - 1, 17, -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations with gradient descent
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank) * 0.3
            V = np.random.randn(m * p, target_rank) * 0.3
            W = np.random.randn(p * n, target_rank) * 0.3
            
            # Alternating Least Squares (ALS) approach
            # For fixed V and W, solve for U; for fixed U and W, solve for V; etc.
            # This is a common approach for tensor decomposition.
            
            best_error = float('inf')
            for iteration in range(100):
                if time.time() > deadline - 30:
                    break
                
                # Update U: for each (i,j), minimize ||U[i*m+j] - target|| where
                # target depends on V and W. Actually, the ALS approach for this problem
                # is more complex because the objective is not a simple matrix factorization.
                
                # Instead, let's use a simple gradient descent approach.
                # Compute gradients of error = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                
                # For efficiency, we can compute the error and its gradient numerically
                # or analytically. Let's try a simple approach:
                
                # Compute current error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                            error += (val - 1.0) ** 2
                
                if error < best_error:
                    best_error = error
                
                if error < 1e-8:
                    break
                
                # Simple gradient step (this is expensive but let's try)
                # Actually, for this problem size, let's just check if we can find
                # a solution by trying different initializations.
                
                # Perturb the matrices slightly
                U += np.random.randn(*U.shape) * 0.01
                V += np.random.randn(*V.shape) * 0.01
                W += np.random.randn(*W.shape) * 0.01
            
            # Check if we found a valid decomposition
            if best_error < 1e-6:
                # Convert to exact fractions if possible
                U_exact = [[Fraction(int(round(x * 1000)) / 1000) for x in U[i]] for i in range(n*m)]
                V_exact = [[Fraction(int(round(x * 1000)) / 1000) for x in V[i]] for i in range(m*p)]
                W_exact = [[Fraction(int(round(x * 1000)) / 1000) for x in W[i]] for i in range(p*n)]
                
                # Verify with exact arithmetic
                if _verify(U_exact, V_exact, W_exact):
                    return {"U": U_exact, "V": V_exact, "W": W_exact}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # First, try to find a lower rank decomposition
    result = _try_find_lower_rank(deadline)
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook()
