"""
Matrix multiplication 3,3,8 decomposition.
Implements the schoolbook algorithm with a safety deadline guard.
"""
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: ensure we return well before the 120s limit.
    # The schoolbook decomposition is O(1) for fixed dimensions, but we add the guard as required.
    start_time = time.time()
    deadline = 115.0 # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition:
    # The tensor T[i,j,k] = 1 if we compute C[i,k] += A[i,j] * B[j,k]
    # We need U[i*m + j, c], V[j*p + k, c], W[k*n + i, c] such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    # This is satisfied by assigning a unique column c to each (i,j,k) tuple.
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check deadline (unlikely to trigger for fixed 3,3,8 but required by spec)
                if time.time() > deadline:
                    # If we somehow exceeded, return what we have. 
                    # Since this is deterministic and fast, this branch is never taken.
                    return {"U": U, "V": V, "W": W}
                
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
