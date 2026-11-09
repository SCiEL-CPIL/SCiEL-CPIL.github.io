"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, np_ = n*m, m*p, p*n
    
    # Build the target tensor
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a low-rank decomposition
    # Start with a lower bound and increase
    # The minimum possible rank is at least the max flattening rank
    
    # Compute flattening ranks
    T1 = T.reshape(nm, mp*np_)
    T2 = T.reshape(mp, nm*np_)
    T3 = T.reshape(np_, nm*mp)
    
    rank1 = np.linalg.matrix_rank(T1)
    rank2 = np.linalg.matrix_rank(T2)
    rank3 = np.linalg.matrix_rank(T3)
    
    lower_bound = max(rank1, rank2, rank3)
    
    # Try ranks from lower_bound up to nm*mp*np_ (but that's too large)
    # For this problem, try a reasonable range
    best_result = None
    best_rank = nm * mp * np_  # schoolbook
    
    # Try a few candidate ranks
    for rank in range(lower_bound, min(lower_bound + 50, nm*mp*np_ + 1)):
        # Try to find a decomposition of this rank
        # This is a hard problem, so use a heuristic
        
        # For now, just try the schoolbook if rank == nm*mp*np_
        if rank == n * m * p:
            U = [[0] * rank for _ in range(nm)]
            V = [[0] * rank for _ in range(mp)]
            W = [[0] * rank for _ in range(np_)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][column] = 1
                        V[j*p+k][column] = 1
                        W[k*n+i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
    
    # If we get here, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(np_)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
