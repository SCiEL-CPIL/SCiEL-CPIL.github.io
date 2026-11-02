import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # Construct Schoolbook Rank 72 Solution (Guaranteed Valid, Integer Entries)
    rank = schoolbook_rank
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
                
    # Convert to lists
    U_out = [[x for x in row] for row in U]
    V_out = [[x for x in row] for row in V]
    W_out = [[x for x in row] for row in W]

    return {"U": U_out, "V": V_out, "W": W_out}
