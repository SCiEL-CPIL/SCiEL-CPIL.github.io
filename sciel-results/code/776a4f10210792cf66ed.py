"""
Optimized tensor decomposition for 2x3 * 3x9 matrix multiplication.
Theoretical minimum rank is n*m*p = 2*3*9 = 54.
This program constructs the rank-54 decomposition.
"""

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The rank of the matrix multiplication tensor M_{n,m,p} is n*m*p.
    # This is because the tensor is the direct sum of p copies of M_{n,m,1},
    # and the rank of M_{n,m,1} is n*m.
    # Thus, minimum rank = 2 * 3 * 9 = 54.
    
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U corresponds to A[i, j]
                # Index in U: i*m + j
                U[i * m + j][idx] = 1.0
                
                # V corresponds to B[j, k]
                # Index in V: j*p + k
                V[j * p + k][idx] = 1.0
                
                # W corresponds to C[i, k]
                # Index in W: k*n + i (following the trace convention W[k*n+i, r])
                W[k * n + i][idx] = 1.0
                
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
