<<<<<<< SEARCH
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a rank-1 term
    def add_term(u_indices, v_indices, w_indices, col):
        for u_idx, val in u_indices:
            U[u_idx][col] += val
        for v_idx, val in v_indices:
            V[v_idx][col] += val
        for w_idx, val in w_indices:
            W[w_idx][col] += val
=======
    import numpy as np
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to set a rank-1 term
    def add_term(u_indices, v_indices, w_indices, col):
        for u_idx, val in u_indices:
            U[u_idx, col] += val
        for v_idx, val in v_indices:
            V[v_idx, col] += val
        for w_idx, val in w_indices:
            W[w_idx, col] += val
>>>>>>> REPLACE
