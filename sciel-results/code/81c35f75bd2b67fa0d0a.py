"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    def _build_schoolbook():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    def _verify(U, V, W, rank):
        """Check if U,V,W of given rank exactly decompose the tensor."""
        # Tensor T[i,j,k] should equal sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Check a few random points
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    U, V, W = _build_schoolbook()
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    deadline = time.time() + 110  # Safety margin
    
    # Try Strassen-based tiling for 4x4 x 4x6
    # Split A into 2x2 blocks of size 2x2
    # Split B into 2x3 blocks of size 2x2 (but 6 = 2+2+2, so 3 blocks of width 2)
    # Actually B is 4x6: 2 block-rows of height 2, 3 block-cols of width 2
    # C is 4x6: same structure
    
    # For each pair of block-rows in A and block-col in B:
    # C[i_block, j_block] = A[i_block,0]*B[0,j_block] + A[i_block,1]*B[1,j_block]
    
    # This is a 2x2 x 2x2 x 2x2 problem tiled 1 x 3 times (3 block-columns)
    # Strassen on 2x2x2x2x2x2: rank 7 per tile instead of 8
    # Total: 3 tiles * 7 = 21 rank-1 terms for the block structure
    # But each "term" involves 2x2x2 = 8 scalar multiplications
    
    # Let me construct this explicitly:
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B02 (rows 0-1, cols 4-5), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3), B12 (rows 2-3, cols 4-5)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3), C02 (rows 0-1, cols 4-5), C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3), C12 (rows 2-3, cols 4-5)
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # Using Strassen on the "2x2 matrix" of blocks for each column j:
    # For column j (j=0,1,2):
    #   M1 = (A0j + A1j) * (B0j + B1j)  -- but this doesn't work directly
    #   
    # Actually, for computing C[0,j] = A00*B0j + A01*B1j and C[1,j] = A10*B0j + A11*B1j:
    # This is a 2x2 (block) matrix product where A is 2x2 blocks and B is 2x1 blocks... 
    # No, it's not square.
    
    # Let me think differently. The full problem is rank-96. 
    # A better approach: use random search with verification.
    
    # Try to find a lower rank by random perturbation and rounding
    import random
    random.seed(42)
    
    # Try various ranks from a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(16, 24, 24) = 24
    # But for matrix mult, known lower bounds are higher.
    # For 4x4x4, best known is 49 (Lader). For 4x4x6, let's try from 48 up.
    
    # Actually, let's try a smarter approach: use the fact that we can express
    # the tensor as a sum of rank-1 tensors and search for sparse solutions.
    
    # For now, return schoolbook as safe baseline
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
