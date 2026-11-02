import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # 10 seconds is more than enough for this trivial computation
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Schoolbook term: A[i,j] * B[j,k] contributes to C[i,k]
                # Index in U: i*m + j
                # Index in V: j*p + k
                # Index in W: k*n + i
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
                if time.time() > deadline:
                    # Safety check, though this loop is tiny
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Convert to lists of lists to match expected interface if strict, 
    # but numpy arrays are often accepted or converted by evaluators. 
    # The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # The seed used lists. I will return lists to be safe.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
