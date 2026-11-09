import time
import sys

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m
    
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indexing:
    # U row: i*m + j
    # V row: j*p + k
    # W row: k*n + i
    
    # Decomposition:
    # For each i in 0..n-1, r in 0..m-1:
    # col = i*m + r
    # U[i*m + r, col] = 1
    # V[r*p + k, col] = 1 for all k?
    # Wait, v_{i,r}(y,z) = delta_{y,r}.
    # So V row index is y*p + z.
    # V[y*p + z, col] = 1 if y == r.
    
    # W row index is k*n + i.
    # w_{i,r}(x,z) = delta_{x,i}.
    # So W[k*n + i, col] = 1.
    
    for i in range(n):
        for r in range(m):
            col = i * m + r
            
            # Fill U
            # U row i*m + r is 1
            U[i * m + r][col] = 1
            
            # Fill V
            # V rows r*p + k for k in 0..p-1 are 1
            for k in range(p):
                V[r * p + k][col] = 1
                
            # Fill W
            # W rows k*n + i for k in 0..p-1 are 1
            for k in range(p):
                W[k * n + i][col] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
