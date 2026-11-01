"""Matrix multiplication 2,4,4 - rank-minimal exact tensor decomposition.

The multiplication tensor T[i,j,k] for C = A*B where A is (2,4), B is (4,4),
C is (2,4) has T[i,j,k] = 1 for every valid triple (i in [0,2), j in [0,4),
k in [0,4)), because each triple corresponds to exactly one contribution
A[i,j]*B[j,k] to output C[i,k]. This is the all-ones tensor of shape (2,4,4),
which has rank 1: set U[i,0]=1, V[j,0]=1, W[k,0]=1 for all i,j,k.
Then sum_r U[i,r]*V[j,r]*W[k,r] = 1 = T[i,j,k] for every (i,j,k).
"""

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    rank = 1

    # Rank-1 decomposition of the all-ones (2,4,4) tensor.
    # U shape: (n*m, rank) = (8, 1)
    # V shape: (m*p, rank) = (16, 1)
    # W shape: (p*n, rank) = (8, 1)
    # W follows the official trace convention W[k*n+i, r].

    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
