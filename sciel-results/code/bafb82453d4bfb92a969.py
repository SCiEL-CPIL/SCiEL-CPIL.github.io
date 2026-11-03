import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    deadline = time.time() + 100 # 100s limit, 20s margin
    
    # Fallback to schoolbook
    def get_schoolbook():
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
        
    best_result = get_schoolbook()
    
    # Search for lower rank
    # Try ranks from 53 down to 40
    for R in range(rank_schoolbook - 1, 39, -1):
        if time.time() > deadline:
            break
            
        # Try a few random instances
        # Using small integers for U, V increases chance of integer W
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Random U, V with entries in {-1, 0, 1}
            # Using 0/1 might be better for sparse solutions
            U = np.random.randint(0, 2, (n * m, R))
            V = np.random.randint(0, 2, (m * p, R))
            
            W = np.zeros((p * n, R))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build M matrix: M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    M = np.zeros((m, R))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve M @ W_vec = ones(m)
                    # W_vec is W[k*n+i, :]
                    try:
                        # Use pinv for least squares
                        W_vec, residuals, rank_M, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        
                        # Check if W_vec is close to integer/half-integer
                        # The problem allows half-integers.
                        W_half = np.round(W_vec * 2) / 2
                        
                        # Verify the solution
                        # M @ W_half should be close to ones(m)
                        if np.allclose(M @ W_half, np.ones(m), atol=1e-9):
                            W[k * n + i, :] = W_half
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                        
            if valid:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                # Ensure they are simple floats/ints
                return {"U": U_list, "V": V_list, "W": W_list}
                
    return best_result
