"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Kronecker-product-style decomposition:
    T_{2,3,10} = T_{2,1,1} ⊗ T_{1,3,10}
    Rank = rank(T_{2,1,1}) * rank(T_{1,3,10}) = 2 * 30 = 60.
    
    This is equivalent to schoolbook but structured as a Kronecker product,
    which makes the algebraic structure explicit and potentially allows
    verification of optimality via the slice-rank lower bound.
    """
    n, m, p = DIMENSIONS
    # rank = n * (m * p) = 2 * 30 = 60
    rank = n * m * p
    
    # U: shape (n*m, rank) = (6, 60)
    # V: shape (m*p, rank) = (30, 60)  
    # W: shape (p*n, rank) = (20, 60)
    # Indexing: U[i*m + j, col], V[j*p + k, col], W[k*n + i, col]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
