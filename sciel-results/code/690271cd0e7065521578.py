import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 37
    # Try to find a decomposition of rank target_rank
    # If it fails, try higher ranks
    
    best_result = None
    start_time = time.time()
    
    # The tensor T is all ones
    T = np.ones((n, m, p))
    
    # We want to find U (n*m, r), V (m*p, r), W (p*n, r)
    # such that sum_c U[:, c] V[:, c] W[:, c] = T (flattened)
    
    # Flatten T
    T_flat = T.flatten()
    
    for r in range(target_rank, 46):
        # Try a few random initializations
        for attempt in range(20):
            if time.time() - start_time > 100:
                break
                
            # Random U, V with small integer entries
            U = np.random.randint(-1, 2, size=(n*m, r))
            V = np.random.randint(-1, 2, size=(m*p, r))
            
            # Construct the linear system for W
            # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i, j, k]
            # For each c, the coefficient of W[k*n+i, c] is U[i*m+j, c] * V[j*p+k, c]
            
            # We have 45 equations and 15*r variables
            A = np.zeros((n*m*p, p*n*r))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for c in range(r):
                            w_idx = c * (p*n) + k * n + i
                            A[eq_idx, w_idx] = U[i*m+j, c] * V[j*p+k, c]
            
            # Solve A * W_flat = T_flat
            # Use least squares
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A, T_flat, rcond=None)
                # Check if the solution is exact
                if np.allclose(A @ W_flat, T_flat):
                    W = W_flat.reshape((p*n, r))
                    # Check if W can be made integer/half-integer
                    # For now, assume we can scale
                    # If we need strict integer, we might need to adjust U, V
                    # But the problem allows half-integers
                    # If W is rational, we can scale
                    # Let's just return this if it's close to integer
                    if np.allclose(W * 2, np.round(W * 2)):
                        best_result = {"U": U, "V": V, "W": W}
                        return best_result
            except:
                pass
        
        if best_result:
            return best_result
            
    # If no better rank found, return schoolbook
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
