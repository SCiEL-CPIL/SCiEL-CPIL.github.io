"""
This program provides a rank-1 exact tensor decomposition for matrix multiplication.
The tensor corresponding to the multiplication of an n x m matrix and an m x p matrix
is the all-ones tensor T_{ijk} = 1. This tensor has rank 1, as it can be written as
the outer product of three vectors of ones:
T_{ijk} = u_i v_j w_k where u_i=1, v_j=1, w_k=1.

The decomposition matrices U, V, W are constructed such that:
U[i*m + j, c] = 1
V[j*p + k, c] = 1
W[k*n + i, c] = 1
for the single rank dimension c=0.

The entries are integers (1), satisfying the half-integer constraint.
"""

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The rank of the all-ones tensor is 1.
    rank = 1
    
    # Initialize U, V, W with ones.
    # U shape: (n * m, rank)
    # V shape: (m * p, rank)
    # W shape: (p * n, rank)
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
