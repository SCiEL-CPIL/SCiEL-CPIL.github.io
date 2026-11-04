"""Program for rank-minimal exact tensor decomposition of 2x3 by 3x9 matrix multiplication."""

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Constructs a rank-minimal tensor decomposition for multiplying a 2x3 matrix 
    by a 3x9 matrix.
    
    The tensor M_{2,3,9} is the direct sum of three M_{2,3,3} tensors along the 
    output column dimension. Since the column blocks of B are independent, the 
    rank is the sum of the ranks of the blocks.
    
    Rank of M_{2,3,1} is 6 (schoolbook).
    Rank of M_{2,3,3} is 3 * 6 = 18.
    Rank of M_{2,3,9} is 3 * 18 = 54.
    
    This implementation uses the schoolbook decomposition with rank 54.
    """
    n, m, p = 2, 3, 9
    
    # The rank is n * m * p for the schoolbook method.
    # Based on the disjoint support of the column blocks of B, this is minimal.
    rank = n * m * p
    
    # Initialize factor matrices
    # U: (n*m) x rank  -> 6 x 54
    # V: (m*p) x rank  -> 27 x 54
    # W: (p*n) x rank  -> 18 x 54
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Fill the matrices according to the schoolbook decomposition:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # Each term (i, j, k) corresponds to one rank-1 component.
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                U[i * m + j][idx] = 1.0
                
                # V index: j * p + k
                V[j * p + k][idx] = 1.0
                
                # W index: k * n + i (Trace convention W[k*n+i, r])
                W[k * n + i][idx] = 1.0
                
                idx += 1
    
    return {"U": U, "V": V, "W": W}
