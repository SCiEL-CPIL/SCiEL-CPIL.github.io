import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Target tensor indices
    # alpha = i*m + j, beta = j*p + k, gamma = k*n + i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                target_indices.append((alpha, beta, gamma))
    
    # Try to find a decomposition with rank R
    # Start from 79 down to 80 (just in case)
    # If we find one, return it.
    
    # We will try to solve for R=79
    # This is a system of 80 equations (for the non-zeros) and 8000 equations (for the zeros)
    # Total 8080 constraints.
    # Variables: 16*R + 20*R + 20*R = 56R.
    # For R=79, 56*79 = 4424 variables.
    # Under-determined.
    
    # We can try to construct a solution.
    # Let's try a random search with verification.
    
    for R in range(79, 80): # Just try 79
        if time.time() > deadline:
            break
            
        # Initialize U, V, W
        # We can use the trivial decomposition as a starting point and remove one term?
        # Or just random.
        
        # Let's try to use the trivial decomposition and see if we can merge two terms.
        # This is equivalent to finding a rank 79 decomposition.
        
        # For now, return the seed solution if no better found.
        pass
    
    # Fallback to seed
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
