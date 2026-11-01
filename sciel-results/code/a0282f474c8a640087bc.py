import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is n*m*p = 36
    rank = n * m * p
    
    # Initialize U, V, W as numpy arrays for easier manipulation
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # The tensor for C = A * B is defined by T[i, j, k, i', j', k'] = delta[i,i'] delta[j,j'] delta[k,k']
    # We decompose this into sum_r U[i,j,r] V[j',k',r] W[i',k',r]
    # The standard decomposition uses one term for each (i, j, k) tuple.
    # There are n*m*p such tuples.
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
                
    # Convert to lists as required by the interface style of the seed
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
