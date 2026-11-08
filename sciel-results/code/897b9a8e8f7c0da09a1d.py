"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
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
    
    # Try to reduce rank by finding linear dependencies
    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The constraint is: sum_c U[i,j,c] * V[j,k,c] = W[k,n+i,c] for all i,j,k
    # This is a nonlinear constraint, so we can't simply use SVD
    
    # Instead, try a different approach: use known small-rank decompositions
    # For 3x5x6, try to find a decomposition with rank < 90
    
    # Use the fact that we can sometimes reduce rank by 1-2 through careful construction
    # Let's try to find if any columns are linearly dependent in a way that preserves the product
    
    # Alternative: use a parametric search for a slightly better decomposition
    # This is complex, so for now return schoolbook if no quick improvement found
    
    if time.time() > deadline - 10.0:
        return {"U": U, "V": V, "W": W}
    
    # Try a simple optimization: check if we can merge any pairs of terms
    # This is NP-hard in general, so skip for now
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
