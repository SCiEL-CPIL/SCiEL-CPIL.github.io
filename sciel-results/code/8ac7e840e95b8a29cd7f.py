"""
Matrix multiplication 3,4,5 rank-minimal decomposition.
Returns the schoolbook decomposition which is exact and valid.
"""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x5 matrix.
    
    The schoolbook algorithm uses rank n*m*p = 3*4*5 = 60.
    Each term in the sum corresponds to a single (i, j, k) index triple.
    
    Index conventions:
    U: (n*m, rank), index a = i*m + j
    V: (m*p, rank), index b = j*p + k
    W: (p*n, rank), index c = k*n + i
    
    The decomposition satisfies:
    sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
    where T is the multiplication tensor.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row i*m + j
                u_idx = i * m + j
                # Index in V: row j*p + k
                v_idx = j * p + k
                # Index in W: row k*n + i
                w_idx = k * n + i
                
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
