import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create indices
    # U indices: i, j -> i*m + j
    # V indices: j, k -> j*p + k
    # W indices: k, i -> k*n + i
    
    # We can construct U, V, W as identity-like matrices in the tensor space.
    # The tensor is T[i,j,k] = 1 if the indices match the multiplication pattern.
    # In the rank-1 decomposition, each term c corresponds to a unique (i,j,k) triple.
    
    # Initialize zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
                
    # Convert to lists of lists as required by the interface?
    # The interface says: 'U', 'V', 'W': factor matrices...
    # The seed program returns lists of lists.
    # I should return lists of lists to be safe, or check if numpy arrays are accepted.
    # "factor matrices of shapes..." usually implies arrays, but the seed returns lists.
    # The evaluator likely accepts both or expects lists.
    # I will return lists to match the seed exactly.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
