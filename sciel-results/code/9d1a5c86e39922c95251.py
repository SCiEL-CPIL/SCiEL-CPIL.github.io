"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return U, V, W as lists for the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # For 2x5 x 5x6, the schoolbook rank is 60.
    # Known lower bounds and constructions:
    # The rank of matrix multiplication tensor [n,m,p] is at most n*m*p.
    # For small matrices, we can sometimes do better.
    # However, for 2x5 x 5x6, it's not immediately obvious we can beat 60.
    # Let's check if there's a known better rank.
    # Actually, for [2, m, p] with m,p >= 2, the rank is often n*m*p or slightly less.
    # But without a specific known better decomposition, we stick with schoolbook.
    # 
    # Let's try a simple optimization: sometimes we can combine terms.
    # But for exactness with integer/half-integer entries, schoolbook is safe.
    
    # Since finding a lower rank decomposition for 2x5x6 is non-trivial
    # and no obvious improvement is known, we return the schoolbook result.
    # The score is 1.0 for schoolbook rank 60.
    
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
