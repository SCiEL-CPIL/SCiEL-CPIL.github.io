"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    
    # Start with the naive rank-1 decomposition
    # Each rank-1 term is indexed by (i, j, k)
    # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    
    # Try to find a lower rank decomposition by exploiting structure
    # Use a greedy approach: start with all 60 terms and try to merge
    
    # For 2x5 * 5x6, try splitting the middle dimension
    # Split m=5 into m1=2, m2=3
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x3
    # B = [B1; B2] where B1 is 2x6, B2 is 3x6
    # AB = A1*B1 + A2*B2
    
    # Rank of A1*B1 (2x2 * 2x6): naive rank = 2*2*6 = 24
    # Rank of A2*B2 (2x3 * 3x6): naive rank = 2*3*6 = 36
    # Total = 60 (no improvement)
    
    # Try a different approach: use the fact that for small n, we can
    # find explicit lower rank decompositions
    
    # For 2x5 * 5x6, let's try to find a rank-58 or lower decomposition
    # by looking for linear dependencies in the rank-1 tensors
    
    # Generate all naive rank-1 terms
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                terms.append((u, v, w))
    
    # Try to find a subset of these that can be combined
    # with integer coefficients to reduce the rank
    
    # For now, use the naive decomposition but check if we can do better
    # by exploiting the specific structure of 2x5 * 5x6
    
    # Known result: rank(2x5 * 5x6) <= 58 via explicit construction
    # Let me try to find such a construction
    
    # Alternative: use the fact that we can write the multiplication
    # as a sum of Kronecker products with shared factors
    
    # Let's try a block decomposition approach
    # Split the 5x5 middle into blocks and use Kronecker structure
    
    # For a 2x5 matrix, we can think of it as 2 blocks of sizes 2x2 and 2x3
    # For a 5x6 matrix, as 2 blocks of sizes 2x6 and 3x6
    
    # The product is A1*B1 + A2*B2
    
    # Now, can we decompose A1*B1 with rank < 24?
    # 2x2 * 2x6: this is equivalent to multiplying a 2x2 by a 2x6
    # The tensor rank of 2x2*2x6 is at most 2*2*6=24, but might be lower
    
    # Actually, for 2x2 * 2x6, we can use the fact that it's like
    # doing two independent 2x2*2x3 multiplications (split the 6 into 3+3)
    # Each 2x2*2x3 has rank at most 2*2*3=12, total 24
    
    # Let me try a more direct approach: just use the naive rank and
    # see if the evaluator accepts it, then iterate
    
    # For now, return the naive decomposition
    rank = n * m * p
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
