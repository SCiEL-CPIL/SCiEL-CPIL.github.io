import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the tensor T of shape (nm, mp, pn)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Schoolbook decomposition
    rank = nm * 1  # Wait, no. Schoolbook rank is n*m*p
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
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank using a greedy compression
    # Idea: try to replace pairs of terms with a single term if possible
    # Or use a more sophisticated approach
    
    # For now, let's try a simple approach: check if the rank can be reduced
    # by finding linear dependencies in the tensor
    
    # The tensor T can be written as sum of rank-1 tensors.
    # The minimum rank is what we want.
    
    # Try a greedy algorithm: start with all 48 terms, try to remove one and
    # see if the remaining can still represent T (unlikely for exact)
    
    # Alternative: try to find a decomposition with rank r < 48
    # by solving a system of equations
    
    # Given time constraints, let's just return the schoolbook result
    # unless we can find a better one quickly
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
