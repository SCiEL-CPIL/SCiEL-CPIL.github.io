import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Try ranks from 22 up to 40
    # The hint suggests 22 is likely minimal.
    for R in range(22, 41):
        if time.time() > deadline:
            break
            
        # Try to find a decomposition of rank R
        # We use random initialization for U and V
        # And solve for W
        # We check if W has half-integer entries
        
        # To increase success rate, we try multiple seeds
        # But we must be careful with time
        max_seeds = 500
        if R > 25:
            max_seeds = 100
            
        for seed in range(max_seeds):
            if time.time() > deadline:
                break
                
            np.random.seed(seed + R * 1000)
            # Use small integers for U and V to keep W small
            # -1, 0, 1
            A = np.random.randint(-1, 2, size=(R, n*m))
            B = np.random.randint(-1, 2, size=(R, m*p))
            
            W = np.zeros((R, p*n))
            success = True
            
            # Pre-compute M matrices for efficiency?
            # M depends on i, j
            # M[idx, r] = A[r, i*m + k] * B[r, kp*p + j]
            
            for i in range(n):
                for j in range(p):
                    M = np.zeros((m*m, R))
                    for k in range(m):
                        for kp in range(m):
                            idx = k * m + kp
                            M[idx, :] = A[:, i*m + k] * B[:, kp*p + j]
                    
                    b = np.eye(m).flatten()
                    
                    # Solve least squares
                    try:
                        w, residuals, rank2, s = np.linalg.lstsq(M, b, rcond=None)
                    except:
                        success = False
                        break
                    
                    # Check if solution is exact
                    if not np.allclose(M @ w, b, atol=1e-9):
                        success = False
                        break
                    
                    # Check if w is close to half-integers
                    w_check = w * 2
                    if not np.allclose(w_check, np.round(w_check), atol=1e-9):
                        success = False
                        break
                        
                    W[:, j*n + i] = w
            
            if success:
                return {"U": A.tolist(), "V": B.tolist(), "W": W.tolist()}
                
    # Fallback to rank 63
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
