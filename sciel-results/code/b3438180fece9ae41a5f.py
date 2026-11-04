import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100
    
    # Try ranks from a low bound upwards
    # Lower bound for rank is max(n*m, m*p, p*n) = max(6, 27, 18) = 27?
    # Actually, rank >= max(n*m, m*p, p*n) is not true.
    # Rank >= max(n, m, p) is not true.
    # Rank >= max(n*m, m*p, p*n) / something?
    # For square 2x2, rank is 7 > 4.
    # For 2x3x9, let's start from 21.
    
    for R in range(21, 55):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random initializations
        for attempt in range(500):
            if time.time() > deadline:
                break
                
            # Random U, V with small integers
            U = np.random.randint(-1, 2, size=(n*m, R))
            V = np.random.randint(-1, 2, size=(m*p, R))
            
            # Check if we can solve for W
            # Reshape U, V for easier access
            # U[i, j, r] where index is i*m+j
            # V[j, k, r] where index is j*p+k
            
            valid_W = True
            W = np.zeros((p*n, R))
            
            for k in range(p):
                for i in range(n):
                    # Construct matrix M_{k,i} of size m x R
                    # M_{k,i}[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    M_ki = np.zeros((m, R))
                    for j in range(m):
                        M_ki[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    # Solve M_ki * w = 1
                    # w is vector of size R
                    # We want w to be small integers/half-integers
                    # Use lstsq
                    try:
                        w, residuals, rank, sv = np.linalg.lstsq(M_ki, np.ones(m), rcond=None)
                    except:
                        valid_W = False
                        break
                    
                    # Check if w is close to half-integers
                    # w * 2 should be close to integers
                    w2 = w * 2
                    w2_rounded = np.round(w2)
                    if np.max(np.abs(w2 - w2_rounded)) < 1e-6:
                        # Snap to half-integers
                        W[k*n+i, :] = w2_rounded / 2
                    else:
                        # It's not close to half-integers. 
                        # We could try to find a different solution in the null space, 
                        # but for simplicity, let's reject this U,V pair.
                        valid_W = False
                        break
                
            if not valid_W:
                continue
                
            # Verify the decomposition exactly
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(R):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                found = True
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
        if found:
            break
            
    # Fallback to schoolbook
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
