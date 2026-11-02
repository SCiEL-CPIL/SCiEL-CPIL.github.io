"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    dim_u = n * m  # 4
    dim_v = m * p  # 12
    dim_w = p * n  # 12
    
    # Build list of nonzero tensor entries
    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_set.add((a, b, c))
    
    # Rank-24 schoolbook decomposition
    rank = 24
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    # Verify correctness
    for a in range(dim_u):
        for b in range(dim_v):
            for c in range(dim_w):
                expected = 1 if (a, b, c) in nonzero_set else 0
                actual = 0.0
                for r in range(rank):
                    actual += U[a, r] * V[b, r] * W[c, r]
                if abs(actual - expected) > 1e-10:
                    raise ValueError(f"Decomposition error at ({a},{b},{c}): got {actual}, expected {expected}")
    
    # Convert to lists of integers
    def to_list(mat):
        return [[int(round(val)) for val in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
