"""Program for 3x5x7 matrix multiplication tensor decomposition using block-circulant embedding."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Strategy: Embed the 3x5x7 multiplication into a larger structure using
    # cyclic convolution. The key insight is that we can use a circulant embedding
    # with size N >= m + p - 1 = 11 to avoid wraparound.
    # 
    # For each row i of A (size n=3), the product (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # can be viewed as a linear convolution of the row A[i,:] with each column of B.
    # 
    # We use the following approach:
    # 1. Embed each row of A into a cyclic vector of length N
    # 2. Use the DFT of size N to diagonalize the circulant structure
    # 3. This gives N rank-1 terms per row of A
    # 
    # Total rank: n * N = 3 * N
    # 
    # For exact rational arithmetic, we need N such that the DFT entries are rational.
    # This happens when N is a power of 2 (using real/imag parts) or when we use
    # a number-theoretic transform.
    #
    # Alternative: Use the fact that we can decompose using polynomial multiplication.
    # The product of two polynomials of degree < m and < p can be computed using
    # N >= m + p - 1 coefficients. The rank of this operation over the reals is N.
    #
    # Let's use N = 16 (next power of 2 >= 11) and use the real DFT structure.
    # The 16-point DFT can be decomposed into real and imaginary parts, giving
    # 32 real rank-1 terms per row, for a total of 3 * 32 = 96 terms.
    #
    # However, the DFT entries involve cos(2*pi*k/16) and sin(2*pi*k/16), which
    # are not rational. So we need a different approach.
    #
    # Better approach: Use the Kronecker product structure and known low-rank
    # decompositions for small matrix multiplication.
    #
    # Actually, let's use a direct construction based on the fact that:
    # rank(3x5x7) <= rank(4x8x8) - correction terms
    #
    # The 4x8x8 multiplication tensor has rank at most 4*8 = 32 using the
    # circulant embedding with N=8 (since 8 >= 8, no wraparound for 8x8).
    # But we need to project from 4x8x8 to 3x5x7.
    #
    # Let's use the following clean construction:
    # For the 5x7 part, we can use the fact that rank(1x5x7) = 7 (dot product).
    # For the 3x5x7, we can share the 7 terms across the 3 rows if we use
    # a structured decomposition.
    #
    # Simplest correct approach: Use the schoolbook decomposition with rank 105.
    # This is guaranteed to be exact and meet the time limit.
    
    rank = n * m * p  # 105
    
    U = [[0] * rank for _ in range(n * m)]  # shape (15, 105)
    V = [[0] * rank for _ in range(m * p)]  # shape (35, 105)
    W = [[0] * rank for _ in range(p * n)]  # shape (21, 105)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                U[i * m + j][col] = 1
                # V[j*p + k, col] = 1
                V[j * p + k][col] = 1
                # W[k*n + i, col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
