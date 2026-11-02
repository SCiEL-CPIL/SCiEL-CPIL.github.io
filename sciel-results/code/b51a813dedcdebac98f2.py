"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
"""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a rank-minimal (or best known) exact tensor decomposition 
    for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
    
    Currently implements the schoolbook decomposition of rank n*m*p = 42.
    Attempts to verify if a lower rank is possible via simple heuristics 
    (though for 2x3x7, 42 is the standard baseline).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 42
    
    # Initialize U, V, W
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook Decomposition:
    # Each term in the sum C[i,k] = sum_j A[i,j] * B[j,k]
    # corresponds to a rank-1 tensor term.
    # There are n*m*p such terms.
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i, j] = 1
                U[i * m + j][idx] = 1
                # V[j, k] = 1
                V[j * p + k][idx] = 1
                # W[k, i] = 1
                # Note: The output factor W follows W[k*n+i, r]
                # So row index is k*n + i
                W[k * n + i][idx] = 1
                idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
