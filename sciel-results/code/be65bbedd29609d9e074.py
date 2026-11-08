"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    # Rank for 2x4 * 4x5 matrix multiplication
    # Known lower bounds and constructions:
    # For 2x4 * 4x5, the tensor rank is at most 2*4*5 = 40 (schoolbook)
    # But we can do better.
    # 
    # The matrix multiplication tensor T_{n,m,p} has rank at most n*m*p.
    # For specific dimensions, better decompositions are known.
    #
    # For 2x4 * 4x5:
    # We can use the fact that this is equivalent to computing 10 output entries
    # from 8 input entries.
    #
    # Let's try to find a low-rank decomposition.
    # The output C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..1, j in 0..3, k in 0..4.
    #
    # Total outputs: 2*5 = 10
    # Total inputs: 2*4 + 4*5 = 8 + 20 = 28 (but A has 8 entries, B has 20 entries)
    #
    # The bilinear form has 10 terms, each being a sum of 4 products.
    #
    # A known result: for multiplying an n x m matrix by an m x p matrix,
    # the rank is at most n*m*p, but for small cases it can be lower.
    #
    # Let's try rank = 24 or lower. Actually, let me think more carefully.
    #
    # For 2x2 * 2x2, rank is 7 (Strassen).
    # For 2x3 * 3x2, rank is 10 (known).
    # For 2x4 * 4x5, let's try to construct a decomposition.
    #
    # Actually, the simplest correct answer with a reasonable rank:
    # Let's use the schoolbook but see if we can reduce.
    #
    # Schoolbook rank = 2*4*5 = 40.
    #
    # Let me try to find a better construction.
    # Consider that C[i,k] = sum_j A[i,j] B[j,k].
    # 
    # We can write this as a tensor decomposition. The tensor T has indices (i,j) for U, (j,k) for V, (i,k) for W.
    # T[(i,j),(j,k),(i,k)] = delta_{same_j} ... no, that's not right.
    #
    # The matrix multiplication tensor is: T_{a,b,c} where a indexes A, b indexes B, c indexes C.
    # T[(i,j),(j',k),(i',k')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    #
    # A rank-r decomposition gives r terms: T = sum_t U_t ⊗ V_t ⊗ W_t
    # where U_t is a vector in R^{n*m}, V_t in R^{m*p}, W_t in R^{p*n}.
    #
    # For the schoolbook: each term corresponds to a fixed (i,j,k), with U having 1 at (i,j), V at (j,k), W at (k,i).
    #
    # Can we combine terms? For fixed i, we have m*p = 20 terms. For 2 values of i, that's 40.
    #
    # Let's try a rank of 32 or so. Actually, let me just use a known good bound.
    # 
    # For 2x4 * 4x5, I'll use the schoolbook rank of 40 as a safe baseline,
    # but let me try to do better with a simple optimization.
    #
    # Actually, let's just return the schoolbook for now to ensure correctness,
    # then we can optimize.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
