"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a lower-rank decomposition using known results or heuristics
    # For 3x3 * 3x8, the naive rank is 72
    # We'll attempt to use a structured approach to reduce rank
    
    # Start with the naive decomposition and see if we can compress
    # One approach: exploit the structure of matrix multiplication
    
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T has indices (i,j,k) with value delta_{i,k} * delta_{j,j} (in some sense)
    # Actually T[i,j,k] = 1 if we're looking at the (i,j) of A and (j,k) of B
    
    # Let's try a different approach: use the fact that for small matrices,
    # there might be known lower bounds or constructions
    
    # For 3x3 * 3x8, let's try to use a block decomposition
    # Since 8 = 2*4, we can try to exploit this structure
    
    # Actually, let's just use the naive decomposition for now, but with numpy
    # for efficiency, and check if we can do better
    
    # Naive rank
    rank_naive = n * m * p
    
    # Try to find a better decomposition
    # For now, return the naive one
    U = np.zeros((n * m, rank_naive), dtype=int)
    V = np.zeros((m * p, rank_naive), dtype=int)
    W = np.zeros((p * n, rank_naive), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
