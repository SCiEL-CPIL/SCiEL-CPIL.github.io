import random
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Trivial Kronecker Product Decomposition
    # Rank = n * m * p
    rank = n * m * p
    
    # Shapes:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        # Decompose rank index r into i, j, k
        # r = i * (m * p) + j * p + k
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        
        # Set U
        u_idx = i * m + j
        U[u_idx][r] = 1
        
        # Set V
        v_idx = j * p + k
        V[v_idx][r] = 1
        
        # Set W
        w_idx = k * n + i
        W[w_idx][r] = 1
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
