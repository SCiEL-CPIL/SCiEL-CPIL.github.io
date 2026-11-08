"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # The tensor: T[i,j,k] = 1 for the multiplication term
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For all other index combos: sum = 0
    
    # Start with schoolbook and try to reduce
    # Try various ranks from low to high
    
    # Known: schoolbook rank = n*m*p = 144
    # Let's try to find a lower rank decomposition
    
    # Approach: Use random initialization and gradient descent on the tensor residual
    
    # The number of constraints: 
    # Non-zero entries: n*m*p = 144 (must equal 1)
    # Zero entries: (n*m)*(m*p)*(p*n) - n*m*p = 16*36*36 - 144 = 20736 - 144 = 20592 (must equal 0)
    # Total constraints: 20736
    
    # For rank r, we have r*(16+36+36) = r*88 variables
    # We need r*88 >= 20736 for a solution to exist generically, but the structure helps
    
    # Let's try a different approach: exploit the Kronecker product structure
    
    # C = A * B where A is 4x4, B is 4x9
    # We can write A as a sum of rank-1 matrices, but that doesn't directly help
    
    # Better: Use the fact that the multiplication tensor is a Kronecker product
    # T = T_4x4x4 ⊗ T_1x4x9? No, that's not quite right
    
    # Let me try: decompose the 4x4 multiplication using Strassen-like ideas
    # Then apply to each of the 9 columns
    
    # Actually, let's just try to solve the linear system for small r
    
    # For a given rank r, the equations are:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # For other (a,b,c) where not (a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k):
    #   sum_c U[a,c] * V[b,c] * W[c2,c] = 0
    
    # This is a system of polynomial equations. Let's try r = 49 (Strassen rank for 4x4)
    # and see if we can extend it
    
    # For now, let's return the schoolbook decomposition as a fallback
    # and try to improve
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # The random search above is flawed (ignores zero constraints) and may produce invalid JSON.
    # We will return the valid schoolbook decomposition to ensure a non-zero score.
    # To improve score, one would need a correct solver for the tensor decomposition.
    
    best_U = U
    best_V = V
    best_W = W
    best_rank = rank
    
    # Convert to lists of Python floats/ints to ensure JSON compatibility
    U_out = [[0.0] * best_rank for _ in range(n * m)]
    V_out = [[0.0] * best_rank for _ in range(m * p)]
    W_out = [[0.0] * best_rank for _ in range(p * n)]
    
    for i in range(n * m):
        for c in range(best_rank):
            U_out[i][c] = float(best_U[i, c])
    for j in range(m * p):
        for c in range(best_rank):
            V_out[j][c] = float(best_V[j, c])
    for k in range(p * n):
        for c in range(best_rank):
            W_out[k][c] = float(best_W[k, c])
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
