import time
import numpy as np
import random

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # For each rank, try to find U, V such that W can be solved
    for rank in range(20, 41):
        if time.time() - start_time > 130:
            break
        
        found = False
        for trial in range(100):
            if time.time() - start_time > 130:
                break
            
            # Random U and V
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            for r in range(rank):
                for idx in range(n*m):
                    if random.random() < 0.4:
                        U[idx, r] = random.choice([-1, 1])
                for idx in range(m*p):
                    if random.random() < 0.4:
                        V[idx, r] = random.choice([-1, 1])
            
            # Solve for W
            W = np.zeros((p*n, rank))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build the 4 x rank system: A @ w = b
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # b[j] = 1
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    
                    b = np.ones(m)
                    
                    # Solve A @ w = b
                    # Use least squares
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if solution is exact
                    if not np.allclose(A @ w, b, atol=1e-9):
                        valid = False
                        break
                    
                    # Check if w has integer/half-integer entries
                    for r in range(rank):
                        val = w[r]
                        # Check if val is close to an integer or half-integer
                        nearest_int = round(val)
                        nearest_half = round(val * 2) / 2
                        if abs(val - nearest_int) > 1e-6 and abs(val - nearest_half) > 1e-6:
                            valid = False
                            break
                    
                    if not valid:
                        break
                    
                    W[k*n+i, :] = w
            
            if valid and verify(U, V, W):
                found = True
                # Round W to nearest integer/half-integer
                for idx in range(p*n):
                    for r in range(rank):
                        val = W[idx, r]
                        nearest_int = round(val)
                        nearest_half = round(val * 2) / 2
                        if abs(val - nearest_int) < abs(val - nearest_half):
                            W[idx, r] = nearest_int
                        else:
                            W[idx, r] = nearest_half
                break
        
        if found:
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: trivial rank 40
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                U[i*m+j, r] = 1
                V[j*p+k, r] = 1
                W[k*n+i, r] = 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
