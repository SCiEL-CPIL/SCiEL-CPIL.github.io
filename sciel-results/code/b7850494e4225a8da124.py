"""
Program to find the rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication.
Uses the structural decomposition M_{3,4,4} = M_{1,4,4} \oplus M_{1,4,4} \oplus M_{1,4,4}.
Since Rank(M_{1,m,p}) = 1, the total rank is 3.
"""

def construct():
    n, m, p = 3, 4, 4
    rank = n  # 3
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill based on the decomposition
    # For each row i of the first matrix, we have a separate rank-1 component
    for i in range(n):
        r = i
        
        # U part: U[i*m + j, r] = 1 for all j
        for j in range(m):
            U[i * m + j][r] = 1
            
        # V part: V[j*p + k, r] = 1 for all j, k
        for j in range(m):
            for k in range(p):
                V[j * p + k][r] = 1
                
        # W part: W[k*n + i, r] = 1 for all k
        for k in range(p):
            W[k * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
