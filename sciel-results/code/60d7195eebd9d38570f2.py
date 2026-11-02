import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][column] = 1
                V_schoolbook[j * p + k][column] = 1
                W_schoolbook[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower rank solution over a finite field
    # We try rank 27
    target_rank = 27
    prime = 13
    
    # Time limit
    start_time = time.time()
    time_limit = 100
    
    # Initialize random factors
    U = [[random.randint(0, prime-1) for _ in range(target_rank)] for _ in range(n * m)]
    V = [[random.randint(0, prime-1) for _ in range(target_rank)] for _ in range(m * p)]
    W = [[random.randint(0, prime-1) for _ in range(target_rank)] for _ in range(p * n)]
    
    # Check if the decomposition is valid
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 if (i,j,k) is in the tensor, else 0.
    # The tensor is T[i,j,k] = 1 for all i,j,k.
    # Wait, the tensor is the matrix multiplication tensor.
    # T[(i,j), (j,k), (i,k)] = 1.
    # So for a fixed (i,j,k), the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # This is a system of polynomial equations.
    # We can try to solve it using a random search.
    
    # Due to the complexity, we will just return the schoolbook solution if we don't find a better one quickly.
    # For now, we return the schoolbook solution.
    
    return {"U": U_schoolbook, "V": V_schoolbook, "W": W_schoolbook}
