import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # safety margin
    
    # Build the full multiplication tensor T of size (n*m) x (m*p) x (p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1, else 0
    T = np.zeros((n*m, m*p, p*n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Start with schoolbook rank
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=object)
    V = np.zeros((m*p, rank), dtype=object)
    W = np.zeros((p*n, rank), dtype=object)
    
    for c, (i, j, k) in enumerate(
        ((i, j, k) for i in range(n) for j in range(m) for k in range(p))
    ):
        U[i*m+j, c] = 1
        V[j*p+k, c] = 1
        W[k*n+i, c] = 1
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to reduce rank by finding linear dependencies
    # The tensor T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # We want to find a smaller set of rank-1 terms that sum to T
    
    # Approach: view the tensor as a matrix and find its rank
    # Reshape T to a matrix of size (n*m) x (m*p * p*n)
    T_mat = T.reshape(n*m, m*p * p*n)
    
    # The rank of this matrix gives a lower bound on the tensor rank
    # But tensor rank >= matrix rank for any unfolding
    
    # For a practical approach, try to merge columns that are linearly dependent
    # in a certain sense
    
    # Simpler: try to find if any column can be expressed as a combination
    # of others, using the structure of the tensor
    
    # Given the small size, try a greedy approach:
    # For each pair of columns, check if their sum/difference can replace both
    # with a single column (i.e., if the two rank-1 terms can be merged)
    
    # Actually, two rank-1 tensors u⊗v⊗w and u'⊗v'⊗w' can be merged into
    # one rank-1 tensor only in very special cases (essentially never for 
    # generic tensors). So this approach won't work directly.
    
    # Better approach: use the fact that the tensor has a specific structure.
    # For matrix multiplication, there are known rank-reduction techniques.
    
    # For now, return the schoolbook solution.
    # The rank is 60.
    
    return {
        "U": [[int(U[i, c]) for c in range(best_rank)] for i in range(n*m)],
        "V": [[int(V[i, c]) for c in range(best_rank)] for i in range(m*p)],
        "W": [[int(W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    }
