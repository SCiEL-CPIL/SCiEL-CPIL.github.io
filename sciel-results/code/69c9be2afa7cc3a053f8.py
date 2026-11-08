import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    
    Approach:
    1. Decompose the 3x4x5 multiplication into 3 independent 1x4x5 sub-problems (one for each row of A).
    2. Each 1x4x5 sub-problem computes C_i = A_i @ B.
       C_i[k] = sum_j A_i[j] * B[j, k].
    3. The rank of a 1xm x p multiplication tensor is p.
       Here m=4, p=5, so rank is 5.
       The decomposition uses 5 multiplications per row of A.
       Total rank = 3 * 5 = 15.
    4. Construct U, V, W with integer entries (0 or 1) realizing this rank-15 decomposition.
    
    Indices:
    U: (n*m, rank) -> (12, 15)
    V: (m*p, rank) -> (20, 15)
    W: (p*n, rank) -> (15, 15) with W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    
    rank = n * p # 3 * 5 = 15
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for i in range(n):
        for k in range(p):
            r = i * p + k
            
            # U: A[i, j] contributes to all j for this row i
            # U index: i*m + j
            for j in range(m):
                U[i * m + j, r] = 1
                
            # V: B[j, k] contributes to all j for this column k
            # V index: j*p + k
            for j in range(m):
                V[j * p + k, r] = 1
                
            # W: C[i, k] receives the result
            # W index: k*n + i
            W[k * n + i, r] = 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
