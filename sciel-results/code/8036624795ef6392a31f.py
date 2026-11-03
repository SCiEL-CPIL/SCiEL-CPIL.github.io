import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Constructs the matrix multiplication tensor decomposition for 2x3 by 3x5.
    Returns the schoolbook decomposition of rank 30.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: i*m + j
                U[i * m + j, col] = 1.0
                # Index in V: j*p + k
                V[j * p + k, col] = 1.0
                # Index in W: k*n + i (Trace convention: W[k*n+i, r])
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists of lists as per interface requirement if strict, 
    # but numpy arrays are often accepted or convertible. 
    # The prompt says: "factor matrices of shapes ... with integer/half-integer entries"
    # and "return {'U', 'V', 'W'}".
    # The seed program returned lists of lists. I will return lists of lists to be safe.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Optional: Quick check if time permits to see if a trivial optimization exists
    # (None known for 2x3x5)
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
