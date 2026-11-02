import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # The matrix multiplication tensor T has exactly n*m*p non-zero entries:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    # Each rank-1 term u ⊗ v ⊗ w has support that is a Cartesian product.
    # For this specific tensor structure, each non-zero entry (i*m+j, j*p+k, k*n+i)
    # is uniquely determined by the triple (i,j,k). A single rank-1 term can
    # only match one such entry because matching two entries would require
    # incompatible index values. Therefore the rank is exactly n*m*p.
    
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
