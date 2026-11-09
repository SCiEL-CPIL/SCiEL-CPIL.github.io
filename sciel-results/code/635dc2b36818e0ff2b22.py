"""Matrix multiplication 2x3 by 3x9: exact tensor decomposition via block structure."""
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for (2x3) * (3x9) -> (2x9).
    
    Uses the observation that the 3x9 factor splits into three 3x3 blocks.
    For each block b in {0,1,2}, the sub-product is (2x3)*(3x3)->(2x3).
    
    Returns:
        U: list of lists, shape (6, rank)  -- encodes A indices (row i, col j)
        V: list of lists, shape (27, rank) -- encodes B indices (row j, col k)
        W: list of lists, shape (18, rank) -- encodes C indices (col k, row i) via W[k*n+i]
    
    All entries are integers (0 or 1), satisfying the half-integer requirement.
    """
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Build via explicit enumeration of all (i, j, k) triples
    # Grouping by block to make the structure explicit
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for block in range(3):
        for i in range(n):
            for j in range(m):
                for k_local in range(3):
                    k = block * 3 + k_local
                    # A index: i*m + j
                    # B index: j*p + k
                    # C index: k*n + i
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
