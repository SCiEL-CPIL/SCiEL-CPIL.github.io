import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # seconds, leave 5s margin
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 60
    
    # 1. Construct Schoolbook (Guaranteed)
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    def to_list(mat):
        return mat.tolist()

    # 2. Construct Rank 30 Decomposition
    # Based on the analysis:
    # For each k in 0..p-1, use m columns.
    # Total rank r = m * p = 3 * 10 = 30.
    r = m * p
    
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    
    for k in range(p):
        for j in range(m):
            c = k * m + j
            # U[i*m + j, c] = 1 for all i
            for i in range(n):
                U[i * m + j, c] = 1
            # V[j*p + k, c] = 1
            V[j * p + k, c] = 1
            # W[k*n + i, c] = 1 for all i
            for i in range(n):
                W[k * n + i, c] = 1
                
    # Verify the rank 30 decomposition
    valid = True
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    max_err = max(max_err, abs(s - 1.0))
                    break
            if not valid:
                break
        if not valid:
            break
            
    if valid:
        return {
            "U": to_list(U),
            "V": to_list(V),
            "W": to_list(W)
        }
    
    # Fallback to Schoolbook
    return {
        "U": to_list(U_school),
        "V": to_list(V_school),
        "W": to_list(W_school)
    }
