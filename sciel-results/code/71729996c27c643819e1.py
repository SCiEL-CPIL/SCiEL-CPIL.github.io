import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as numpy arrays for efficient construction, then convert
    U_np = np.zeros((n * m, rank))
    V_np = np.zeros((m * p, rank))
    W_np = np.zeros((p * n, rank))
    
    # Schoolbook decomposition
    # Index mapping:
    # U row: i*m + j
    # V row: j*p + k
    # W row: k*n + i
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U_np[u_idx, col] = 1.0
                V_np[v_idx, col] = 1.0
                W_np[w_idx, col] = 1.0
                col += 1
                
    # Convert to lists of lists as required by the interface (implied by seed)
    # The seed returns lists of lists. The evaluator likely accepts numpy arrays too, 
    # but to be safe and distinct, I'll return lists.
    U = U_np.tolist()
    V = V_np.tolist()
    W = W_np.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
