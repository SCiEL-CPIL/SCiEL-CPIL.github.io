import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # The minimal rank for matrix multiplication M_{n,m,p} is m (if we fix the structure 
    # where U and V are independent of i and k respectively, and W is constant).
    # Specifically, I_m = sum_{r=0}^{m-1} e_r e_r^T.
    # This requires R = m.
    # For n=3, m=5, p=7, R=5.
    
    R = m
    
    # Initialize U, V, W with zeros
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Construct the decomposition
    # For each r in 0..m-1:
    # U[i*m + j, r] = 1 if j == r else 0
    # V[j*p + k, r] = 1 if j == r else 0
    # W[k*n + i, r] = 1
    
    for r in range(m):
        # Fill U
        for i in range(n):
            for j in range(m):
                if j == r:
                    U[i * m + j][r] = 1
        
        # Fill V
        for j in range(m):
            for k in range(p):
                if j == r:
                    V[j * p + k][r] = 1
        
        # Fill W
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
