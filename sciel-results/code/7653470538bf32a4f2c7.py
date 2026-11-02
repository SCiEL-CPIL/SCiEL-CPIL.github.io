import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # m is 4, which is 2*2.
    # We decompose m into m1=2, m2=2.
    # A is n x m1 x m2
    # B is m1 x m2 x p
    # C is n x p
    # C[i, k] = sum_{j1, j0} A[i, j1, j0] B[j1, j0, k]
    # = sum_{j1} ( sum_{j0} A[i, j1, j0] B[j1, j0, k] )
    
    # This is a sum of m1 independent n x m2 x p multiplications.
    # Here n=2, m2=2, p=7.
    # So we have 2 independent 2 x 2 x 7 multiplications.
    
    # Rank of 2 x 2 x 7 is 7 * Rank(2 x 2 x 1).
    # Rank(2 x 2 x 1) is 4.
    # So Rank(2 x 2 x 7) is 28.
    # Total rank is 2 * 28 = 56.
    
    # We construct the matrices U, V, W for the full 2 x 4 x 7 problem
    # using this block structure.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Iterate over the block index j1 (0 to 1)
    for j1 in range(2):
        # For each block, we have a 2 x 2 x 7 multiplication.
        # The indices for this sub-problem:
        # i in 0..1
        # j0 in 0..1 (local col of A, local row of B)
        # k in 0..6
        
        # The global indices:
        # A index: i * m + (j1 * 2 + j0)
        # B index: (j1 * 2 + j0) * p + k
        # C index: k * n + i
        
        # We can use the schoolbook decomposition for the sub-problem.
        # Sub-rank is 2 * 2 * 7 = 28.
        # Total rank is 2 * 28 = 56.
        
        for i in range(n):
            for j0 in range(2):
                for k in range(p):
                    c_idx = j1 * (n * 2 * p) + i * (2 * p) + j0 * p + k
                    # This mapping ensures unique index for each (j1, i, j0, k)
                    
                    # Global row in U (A index)
                    u_row = i * m + (j1 * 2 + j0)
                    # Global row in V (B index)
                    v_row = (j1 * 2 + j0) * p + k
                    # Global row in W (C index)
                    w_row = k * n + i
                    
                    U[u_row, c_idx] = 1.0
                    V[v_row, c_idx] = 1.0
                    W[w_row, c_idx] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
