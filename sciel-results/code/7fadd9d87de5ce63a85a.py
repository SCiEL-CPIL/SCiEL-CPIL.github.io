import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Schoolbook rank
    R_school = n * m * p
    U_school = [[0] * R_school for _ in range(n * m)]
    V_school = [[0] * R_school for _ in range(m * p)]
    W_school = [[0] * R_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = R_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find a lower rank decomposition
    # We try ranks R from R_school - 1 down to R_school - 5
    # Using random integer entries in {-1, 0, 1}
    
    for R in range(R_school - 1, max(R_school - 10, 50), -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            W = np.random.randint(-1, 2, size=(p * n, R))
            
            # Check if this decomposes the tensor
            # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # We need T[i,j,k] = 1 for all i,j,k
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_rank = R
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break # Found lower rank, stop searching lower ranks? Or keep going?
                # Let's keep going to find even lower?
                # But 71 is already an improvement.
                
    return {"U": best_U, "V": best_V, "W": best_W}
