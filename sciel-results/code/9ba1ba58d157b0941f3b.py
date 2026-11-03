import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices as lists of lists to match seed interface
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Use a single loop over the rank index, decoding the indices
    # This is distinct from the nested loop structure of the seed
    for t in range(rank):
        # Decode t into i, k, j
        # t = i * (m * p) + k * p + j
        i = t // (m * p)
        rem = t % (m * p)
        k = rem // p
        j = rem % p
        
        # Calculate indices for U, V, W
        u_idx = i * m + k
        v_idx = k * p + j
        w_idx = j * n + i  # Trace convention: W[col_B * n + row_A]
        
        # Set the 1s
        U[u_idx][t] = 1.0
        V[v_idx][t] = 1.0
        W[w_idx][t] = 1.0
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
