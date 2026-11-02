import numpy as np

def construct():
    # Dimensions for 2x3 matrix * 3x3 matrix
    n, m, p = 2, 3, 3
    rank = n * m * p
    
    # Allocate factor matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Construct the standard schoolbook decomposition
    # Each term corresponds to a single multiplication A[i,j] * B[j,k] contributing to C[i,k]
    term_idx = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                # Map logical indices to flat indices
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, term_idx] = 1.0
                V[v_idx, term_idx] = 1.0
                W[w_idx, term_idx] = 1.0
                
                term_idx += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
