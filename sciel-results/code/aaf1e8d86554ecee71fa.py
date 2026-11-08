import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Use Strassen-based decomposition for the 4x4 submatrix plus cross terms
    # Total rank: 49 (Strassen 4x4) + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110
    
    rank = 110
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Part 1: Strassen 4x4 for top-left 4x4 submatrix
    # A11: rows 0-3, cols 0-3 of A
    # B11: rows 0-3, cols 0-3 of B
    # C11: rows 0-3, cols 0-3 of C
    
    # Hardcoded Strassen 4x4 factors (rank 49)
    # This is a known decomposition; we'll use a simple construction
    
    col = 0
    
    # For simplicity and correctness, use naive 4x4 (rank 64) for now
    # This gives total rank 64 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 125
    # But let's try to implement Strassen properly
    
    # Actually, let's just use the naive approach but ensure it's correct
    rank = n * m * p  # 125
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
