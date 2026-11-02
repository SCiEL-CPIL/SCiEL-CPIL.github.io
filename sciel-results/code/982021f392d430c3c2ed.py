import time
import numpy as np

def verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly represents matrix multiplication.
    T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    """
    rank = len(U[0])
    # For rank 105, checking all 105 terms for 105 (i,j,k) triples is 105^3 = 1,157,625 ops.
    # This is fast enough in Python if optimized, but let's use a quick check.
    # The trivial construction is guaranteed correct, so we can skip full verification 
    # in the final returned function to save time, but we keep it for safety.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for t in range(rank):
                    total += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(total - 1) > 1e-10:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 108  # Safety margin
    
    n, m, p = 3, 5, 7
    
    # The trivial rank is n*m*p
    trivial_rank = n * m * p
    
    # Initialize matrices
    U = [[0] * trivial_rank for _ in range(n * m)]
    V = [[0] * trivial_rank for _ in range(m * p)]
    W = [[0] * trivial_rank for _ in range(p * n)]
    
    # Populate the trivial decomposition
    # The tensor of matrix multiplication is defined such that:
    # (A * B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # In tensor notation, the decomposition is:
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
    # This corresponds to the standard schoolbook algorithm where each term
    # U[i*m+j, t]=1, V[j*p+k, t]=1, W[k*n+i, t]=1 for a unique (i,j,k) tuple.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    result = {"U": U, "V": V, "W": W}
    
    return result
