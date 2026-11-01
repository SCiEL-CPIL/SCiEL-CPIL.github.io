"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 4
    M = m * p  # 30
    P = p * n  # 30
    
    deadline = time.time() + 110.0  # Safety margin
    
    # Build the full tensor T of shape (N, M, P)
    T = np.zeros((N, M, P), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1
    
    # Strategy: Use the known structure of the tensor to find a lower rank decomposition.
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 iff i=i', j=j', k=k'.
    # This is a "diagonal" tensor in the sense that it only has 1s where the indices match.
    # For small dimensions, we can try to find a decomposition with rank < n*m*p.
    
    # For 2x2x15, the schoolbook rank is 60.
    # We can try to find a decomposition with rank 59 or lower.
    
    # One approach: use the fact that the tensor is sparse and has a specific pattern.
    # We can try to find a decomposition where some terms are combined.
    
    # For now, let's try a simple approach:
    # 1. Start with the schoolbook decomposition (rank 60).
    # 2. Try to find a lower rank decomposition by searching for patterns.
    
    # Since this is a complex problem, let's try a few specific patterns that might work.
    
    # Pattern 1: Try to combine terms where i and j are the same but k varies.
    # This might not directly work, but let's try.
    
    # For a more systematic approach, we can use the following:
    # The tensor T can be written as a sum of rank-1 tensors.
    # We want to find the minimum number of such tensors.
    
    # For 2x2x15, the minimum rank is known to be less than 60 in some cases.
    # Let's try to find a decomposition with rank 59.
    
    # To find a rank-59 decomposition, we need to find a way to combine two terms.
    # This is equivalent to finding a linear dependence among the 60 rank-1 tensors.
    
    # Let's try a simple search:
    # For each pair of terms (r1, r2), check if their sum can be written as a single rank-1 tensor.
    # This is unlikely, but let's try.
    
    # A better approach: use the structure of the tensor to find a decomposition.
    # The tensor T has the property that T[i*m+j, j'*p+k, k'*n+i'] = 1 iff i=i', j=j', k=k'.
    # This means that for each (i, j, k), there is exactly one non-zero entry.
    
    # We can try to find a decomposition where the terms are grouped by (i, j) or (j, k) or (i, k).
    
    # For example, for fixed i and j, the terms with different k can be combined.
    # Let's try this approach.
    
    # For fixed i and j, the terms are:
    # T[i*m+j, j*p+k, k*n+i] = 1 for k = 0, 1, ..., 14.
    # We can try to write this as a sum of rank-1 tensors.
    
    # Let's try to find a decomposition where for each (i, j), we use a single rank-1 tensor.
    # This would give a rank of 4, which is too optimistic.
    
    # A more realistic approach: use a decomposition where the rank is close to 60 but slightly lower.
    
    # For now, let's return the schoolbook decomposition.
    # In a future iteration, we can try to find a lower rank decomposition.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
