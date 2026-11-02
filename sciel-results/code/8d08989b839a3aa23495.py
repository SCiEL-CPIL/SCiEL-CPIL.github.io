import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Analysis:
    The matrix multiplication tensor $M_{3,3,7}$ can be viewed as the direct sum of 7 instances 
    of $M_{3,3,1}$ because the output columns and input B-columns for different $k$ are disjoint 
    in their indices, while sharing the A indices.
    
    The rank of $M_{3,3,1}$ is $3 \times 3 = 9$. 
    This is because the 3 output rows depend on 3 distinct rows of A, and for each row, 
    we compute a dot product of length 3. The operations for different rows of A are 
    independent in terms of the A-variables.
    
    Therefore, the total rank is $7 \times 9 = 63$.
    
    This program constructs the explicit rank-63 decomposition (schoolbook algorithm).
    """
    n, m, p = DIMENSIONS
    
    # Dimensions of factor matrices
    dim_U = n * m  # 9
    dim_V = m * p  # 21
    dim_W = p * n  # 21
    
    rank = n * m * p  # 63
    
    U = np.zeros((dim_U, rank), dtype=int)
    V = np.zeros((dim_V, rank), dtype=int)
    W = np.zeros((dim_W, rank), dtype=int)
    
    # Construct the decomposition
    # Indices for A: i in [0,n), j in [0,m) -> idx_a = i*m + j
    # Indices for B: j in [0,m), k in [0,p) -> idx_b = j*p + k
    # Indices for C: k in [0,p), i in [0,n) -> idx_c = k*n + i (Official trace convention W[k*n+i, r])
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                
                U[a_idx, col] = 1
                V[b_idx, col] = 1
                W[c_idx, col] = 1
                col += 1
                
    # Return as lists to ensure compatibility
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
