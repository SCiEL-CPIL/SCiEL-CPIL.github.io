import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s limit to be safe
    
    # Try rank 17
    R = 17
    # Flatten U, V, W into a single vector x
    # U: n*m x R = 6 x 17 = 102
    # V: m*p x R = 9 x 17 = 153
    # W: p*n x R = 6 x 17 = 102
    # Total vars: 357
    
    def residuals(x):
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = x[n*m*R + m*p*R:].reshape(p*n, R)
        
        # Calculate sum_r U[i,j,r] V[j,k,r] W[k,i,r] for all i,j,k
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        # We need to compute the tensor product sum efficiently.
        # Let's iterate over r and accumulate.
        # But this is inside the residual function, called many times.
        # We can vectorize.
        
        # U is (n*m, R)
        # V is (m*p, R)
        # W is (p*n, R)
        
        # We want to compute S[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's reshape U to (n, m, R)
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # We need S[i, j, k]
        # S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        # This creates a tensor of size n x m x p = 2 x 3 x 3 = 18 elements.
        
        S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # The target is 1 for all i,j,k
        target = np.ones((n, m, p))
        error = S - target
        return error.flatten()
    
    best_score = n * m * p  # 18
    best_solution = None
    
    # Try a few random restarts
    max_restarts = 50
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Small random values to help convergence
        x0 = np.random.normal(0, 1, size=(n*m + m*p + p*n) * R)
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if result.cost < 1e-8:
                # Found a solution
                # Now try to find a rational approximation
                # Extract U, V, W
                U = result.x[:n*m*R].reshape(n*m, R)
                V = result.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W = result.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Check if we can scale to integers or half-integers
                # We can try to find a common denominator
                # If entries are close to rationals with small denominators, we are good.
                
                # Simple check: if we can round to 0.5 increments and it still works
                # Or just return the float solution if the evaluator accepts floats?
                # The prompt says "integer/half-integer entries".
                # So we MUST return integers or half-integers.
                
                # Let's try to round to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify the rounded solution
                U_3d_r = U_r.reshape(n, m, R)
                V_3d_r = V_r.reshape(m, p, R)
                W_3d_r = W_r.reshape(p, n, R)
                S_r = np.einsum('ijr,jkr,kir->ijk', U_3d_r, V_3d_r, W_3d_r)
                if np.allclose(S_r, np.ones((n, m, p)), atol=1e-9):
                    best_score = R
                    best_solution = (U_r, V_r, W_r)
                    break
        except Exception:
            pass
    
    if best_solution is not None:
        U, V, W = best_solution
        # Convert to lists
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
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
# EVOLVE-BLOCK-END
