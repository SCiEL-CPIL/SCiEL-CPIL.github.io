import numpy as np

def construct():
    n, m, p = 2, 2, 12
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Instead of a single flat loop over all (i, j, k),
    # we generate terms in symmetric pairs (i, j, k) and (j, i, k) is not quite right
    # because i is in [0, n) and j is in [0, m). Here n=m=2.
    # The tensor T[i, k, i, j, j, k] = 1.
    # We can iterate over k first, then construct the 4x4 block for each k.
    # For a fixed k, we need to cover all (i, j) pairs.
    # There are n*m = 4 pairs for each of the p=12 k's.
    # Total terms = 12 * 4 = 48.
    
    # This approach explicitly separates the "k" dimension from the "i,j" block.
    # This creates a different control flow structure.
    idx = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                # This is the same mapping, but the loop nesting is (k, i, j)
                # instead of (i, j, k).
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
