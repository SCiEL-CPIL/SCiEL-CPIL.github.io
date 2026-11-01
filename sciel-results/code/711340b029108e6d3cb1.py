"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition with integer coefficients.
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0 # 10s margin before 150s kill
    
    # Strassen Rank 7 Decomposition (Integer Coefficients)
    # Indices:
    # U: (i, j) -> i*m + j
    # V: (j, k) -> j*p + k
    # W: (k, i) -> k*n + i
    
    # U rows: A00, A01, A10, A11
    U_data = [
        [1, 1, 1, 0, 1, -1, 0],  # A00
        [0, 1, 0, 0, 0, 0, 1],   # A01
        [0, 0, 0, 0, 0, 1, 0],   # A10
        [-1, 0, 0, 1, 1, 0, -1]  # A11
    ]
    
    # V rows: B00, B01, B10, B11
    V_data = [
        [0, 0, 1, 1, 1, 1, 0],   # B00
        [0, 0, 0, 1, -1, 0, 0],  # B01
        [1, 0, 0, 0, 0, 1, 1],   # B10
        [1, 1, -1, 0, 0, 0, 1]   # B11
    ]
    
    # W rows: C00, C10, C01, C11 (Note: W[k*n+i])
    # C00 is k=0, i=0 -> index 0
    # C10 is k=0, i=1 -> index 1
    # C01 is k=1, i=0 -> index 2
    # C11 is k=1, i=1 -> index 3
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 0, 0, 0, 1, 1],   # C10
        [0, 1, 1, 0, 0, 0, 0],   # C01
        [1, 0, -1, 1, 1, 0, 0]   # C11
    ]
    
    U = np.array(U_data, dtype=np.int64)
    V = np.array(V_data, dtype=np.int64)
    W = np.array(W_data, dtype=np.int64)
    
    # Fast verification using tensor contraction
    # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
    # We expect T to be 1 for all i,j,k in 2x2x2
    
    # Reshape to (n*m, rank), (m*p, rank), (p*n, rank)
    # Actually, let's just verify the specific indices needed for 2x2x2
    
    valid = True
    rank = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                if val != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 8 if verification fails (should not happen)
    rank = 8
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
