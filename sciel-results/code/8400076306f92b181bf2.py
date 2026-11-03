import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # The tensor for standard matrix multiplication C = AB is T[i,j,k] = 1 for all i,j,k.
    # This is because C[i,k] = sum_j A[i,j] * B[j,k].
    # The coefficient of A[i,j] * B[j,k] in C[i,k] is 1.
    # Therefore, the tensor is the all-ones tensor.
    # The rank of the all-ones tensor is 1.
    # We can construct it with U, V, W being vectors of 1s.
    
    rank = 1
    
    # Shapes:
    # U: (n*m, rank) -> (15, 1)
    # V: (m*p, rank) -> (25, 1)
    # W: (p*n, rank) -> (15, 1)
    
    # Using lists of lists as in seed
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    # Verify
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For any i,j,k:
    # U[i*m+j, 0] = 1
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    # Product = 1. Sum = 1.
    # This matches T[i,j,k] = 1.
    
    return {"U": U, "V": V, "W": W}
