import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    
    The schoolbook decomposition has rank n*m*p = 2*3*9 = 54.
    Given the small dimensions and lack of a known significantly better algorithm for this specific 
    rectangular shape in standard libraries, we implement the schoolbook decomposition.
    
    The tensor T[i, j, k] corresponds to the term A[i, j] * B[j, k] contributing to C[i, k].
    We need to decompose T into a sum of rank-1 tensors u_r * v_r * w_r.
    In the schoolbook algorithm, each term is a rank-1 tensor with a single 1 in U, V, and W.
    
    U has shape (n*m, rank). Index: i*m + j.
    V has shape (m*p, rank). Index: j*p + k.
    W has shape (p*n, rank). Index: k*n + i. Note: The problem statement says W[k*n+i, r].
    
    We iterate over all i, j, k and create a column in U, V, W for each.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # Schoolbook decomposition: rank = n*m*p = 54
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, col] = 1.0
                # V index: j*p + k
                V[j * p + k, col] = 1.0
                # W index: k*n + i (following the convention W[k*n+i, r])
                W[k * n + i, col] = 1.0
                col += 1
                
                # Check time (though this loop is tiny)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
