"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Returns {'U', 'V', 'W'} with shapes (6, rank), (12, rank), (8, rank).
    U has shape (n*m, rank) = (6, rank)
    V has shape (m*p, rank) = (12, rank) 
    W has shape (p*n, rank) = (8, rank)
    W[k*n+i, r] is the trace convention.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p = 24
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Try to reduce rank using Strassen-like techniques
    # For 2x3x4, we can try to exploit structure
    
    # Strategy: Use the fact that we can combine some terms
    # Let's try a decomposition that uses fewer rank-1 terms
    
    # One approach: treat it as blocks
    # A is 2x3, B is 3x4
    # We can split A into [A1 A2] where A1 is 2x1 and A2 is 2x2
    # And B into [B1; B2; B3] where each is 1x4
    
    # Actually, let me try a direct approach with rank 20 or so
    # The theoretical minimum for 2x3x4 is between 12 and 24
    
    # Let's try to find a good decomposition using random initialization and gradient descent
    # But since we need exact integer/half-integer results, let's be careful
    
    # For now, return the schoolbook which is known to work
    # But let's try to improve it
    
    # Try rank 22 decomposition
    # We can potentially combine 2 terms into 1 using Strassen-like tricks
    
    # Let me try a different approach: enumerate all possible rank-r decompositions
    # This is NP-hard, so let's use a heuristic
    
    # Start with schoolbook and try to merge columns
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to reduce rank by checking if any two columns can be merged
    # (this is a crude heuristic)
    
    # For the time being, let's return the schoolbook decomposition
    # which is guaranteed to be correct
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
