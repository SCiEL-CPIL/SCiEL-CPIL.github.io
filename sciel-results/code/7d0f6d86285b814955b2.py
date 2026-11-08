import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We use the Kronecker product structure based on the factorization of m.
    # m = 4 can be factored as 2 * 2.
    # Let m = m1 * m2 with m1=2, m2=2.
    # A is n x m, reshaped to n x m1 x m2.
    # B is m x p, reshaped to m1 x m2 x p.
    # C is n x p.
    # C[i, k] = sum_{a=0}^{m1-1} sum_{b=0}^{m2-1} A[i, a, b] * B[a, b, k]
    
    # The rank is the sum of ranks of each term.
    # Each term is an outer product of A[:, a, b] (size n) and B[a, b, :] (size p).
    # The rank of the multiplication tensor for n x 1 by 1 x p is n * p.
    # Total rank = m1 * m2 * n * p = m * n * p.
    
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over the decomposed indices of m
    m1 = 2
    m2 = 2
    for a in range(m1):
        for b in range(m2):
            # The index j in the original m dimension is j = a * m2 + b
            j = a * m2 + b
            
            # For this fixed (a, b), we have a term A[:, j] * B[j, :]
            # This corresponds to an outer product of a vector of size n and a vector of size p.
            # The rank of this sub-tensor is n * p.
            for i in range(n):
                for k in range(p):
                    # U index: i * m + j
                    U[i * m + j][col] = 1
                    # V index: j * p + k
                    V[j * p + k][col] = 1
                    # W index: k * n + i
                    W[k * n + i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
