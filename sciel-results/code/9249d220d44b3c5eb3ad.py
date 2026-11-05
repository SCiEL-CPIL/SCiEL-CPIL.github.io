"""Matrix multiplication 3x5x7 - rank-minimal decomposition."""
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 7
    
    # The matrix multiplication tensor T[i,j,k] = 1 if and only if
    # we are multiplying (i,j) of first matrix by (j,k) of second.
    # In the trace convention W[k*n + i, r], the tensor is T[i,j,k].
    
    # We will use a Kronecker product approach:
    # Decompose 3x5x7 as Kronecker of 3x3x3 (rank 23) and 1x5x7 (rank 5*7=35).
    # But this gives rank 23*35 = 805, which is worse than 105.
    
    # Instead, let's try a smarter approach:
    # Split the problem into blocks and use optimal decompositions for each.
    
    # Actually, let's just use the schoolbook decomposition which is guaranteed to work
    # and has rank = n*m*p = 105.
    
    # But we can do better by exploiting the structure.
    
    # Let's try a different approach: use the fact that 3x5x7 can be decomposed
    # using the Strassen-like algorithm for 3x3 matrices, extended to 5x7.
    
    # For now, let's use a simple but valid decomposition.
    # The schoolbook decomposition has rank = n*m*p = 105.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
