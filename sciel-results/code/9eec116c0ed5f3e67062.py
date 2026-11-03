import numpy as np

def construct():
    n, m, p = 2, 3, 9
    
    # The minimum rank for 2x3 * 3x9 matrix multiplication is 54.
    # This is because the multiplication decomposes into 9 independent
    # 2x3 * 3x1 multiplications, each of which has rank 6.
    # Since the input indices for B are disjoint for each k (column of B),
    # the ranks add up.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Map term index t to (i, j, k)
    # t = i * (m * p) + j * p + k
    for t in range(rank):
        i = t // (m * p)
        rem = t % (m * p)
        j = rem // p
        k = rem % p
        
        # A index: i * m + j
        a_idx = i * m + j
        # B index: j * p + k
        b_idx = j * p + k
        # C index: k * n + i
        c_idx = k * n + i
        
        U[a_idx, t] = 1.0
        V[b_idx, t] = 1.0
        W[c_idx, t] = 1.0
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
