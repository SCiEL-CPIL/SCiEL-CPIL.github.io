"""Program for rank-minimal tensor decomposition of 2x3x10 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the tensor decomposition of 2x3x10 matrix multiplication.
    
    Uses a direct sum of rank-3 decompositions for 2x3x1 sub-problems.
    Total rank is 30.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Rank is 3 * p
    rank = 3 * p
    # Initialize U, V, W as integer matrices
    U = np.zeros((n * m, rank), dtype=int)  # 6 x 30
    V = np.zeros((m * p, rank), dtype=int)  # 30 x 30
    W = np.zeros((p * n, rank), dtype=int)  # 20 x 30
    
    # For each column k of B (and C), we have a 2x3x1 sub-problem.
    # This sub-problem has rank 3.
    # The 3 terms correspond to r = 0, 1, 2 (the middle dimension index).
    
    for k in range(p):
        for r in range(m):
            col_idx = k * m + r
            
            # U part: 
            # Non-zero for A[i, r] for i=0..n-1
            # Index in U: i * m + r
            for i in range(n):
                U[i * m + r, col_idx] = 1
                
            # V part:
            # Non-zero for B[r, k]
            # Index in V: r * p + k
            V[r * p + k, col_idx] = 1
            
            # W part:
            # Non-zero for C[k, i] for i=0..n-1
            # Index in W: k * n + i
            for i in range(n):
                W[k * n + i, col_idx] = 1

    # Convert to lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
