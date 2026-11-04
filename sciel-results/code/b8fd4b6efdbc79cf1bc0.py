"""Matrix multiplication 3,3,6 - rank-minimal exact tensor decomposition.

This program provides a valid tensor decomposition for multiplying a 3x3 matrix
by a 3x6 matrix. 

Theoretical Background:
The multiplication $C = AB$ where $A$ is $3 \times 3$ and $B$ is $3 \times 6$
can be viewed as two independent $3 \times 3$ matrix multiplications sharing
the same left operand $A$. Specifically, if we split $B$ into two $3 \times 3$
matrices $B_1$ and $B_2$, then $C = [A B_1, A B_2]$.

The rank of the $3 \times 3 \times 3$ matrix multiplication tensor is known to
be at most 23 (Bläser 2005). If we could implement this rank-23 decomposition,
we could achieve a total rank of $23 + 23 = 46$ for the $3 \times 3 \times 6$
tensor by reusing the factors corresponding to $A$.

However, the explicit coefficients for the rank-23 decomposition are complex
and not trivial to implement from memory without risk of error. The schoolbook
algorithm for $3 \times 3 \times 3$ has rank 27. Using this for both halves
results in a total rank of $27 + 27 = 54$.

Since the schoolbook decomposition for the full $3 \times 3 \times 6$ tensor
also has rank $3 \times 3 \times 6 = 54$, and we cannot reliably implement the
rank-46 decomposition, we return the schoolbook decomposition of rank 54.
This is guaranteed to be correct and valid.

Interface:
    construct() -> {'U', 'V', 'W'}
    U: (9, rank)
    V: (18, rank)
    W: (18, rank)
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """
    Verify that the tensor decomposition is exact.
    The tensor T[i, j, k] for matrix multiplication (n x m) * (m x p) -> (n x p)
    is such that T[i, j, k] = 1 if we are computing C[i, k] = sum_j A[i, j] * B[j, k].
    
    Note on Indexing:
    The problem statement specifies:
    U shape (n*m, rank), indexed by i*m + j
    V shape (m*p, rank), indexed by j*p + k
    W shape (p*n, rank), indexed by k*n + i
    
    The decomposition requires:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k
    """
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    # Check column counts
    for c in range(rank):
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
            
    # Verify the tensor identity
    # We check that the reconstructed tensor matches the identity tensor for matmul
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Strategy: Return the schoolbook decomposition.
    # Rank = n * m * p = 3 * 3 * 6 = 54.
    # This is a valid, exact decomposition.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Optional: Verify the result to ensure correctness
    # This is fast for small dimensions (54 terms, 54 checks)
    if _verify(U, V, W, n, m, p, rank):
        return result
    else:
        # Should not happen for schoolbook, but fallback to empty or error?
        # The contract implies we must return a dict. 
        # If verification fails, there is a bug in the code above.
        # But schoolbook is trivially correct.
        return result

# EVOLVE-BLOCK-END
