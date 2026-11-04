import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # We'll build the decomposition block by block
    # U: (n*m, rank) = (20, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (24, rank)
    
    # Split:
    # A (4x5): rows 1-3 (top), row 4 (bot)
    #          cols 1-3 (left), cols 4-5 (right)
    # B (5x6): rows 1-3 (top), rows 4-5 (bot)
    #          cols 1-3 (left), cols 4-6 (right)
    
    # Products:
    # 1. A[0:3, 0:3] (3x3) * B[0:3, 0:3] (3x3) -> C[0:3, 0:3] (3x3)
    # 2. A[0:3, 3:5] (3x2) * B[3:5, 0:3] (2x3) -> C[0:3, 0:3] (3x3)
    # 3. A[0:3, 0:3] (3x3) * B[0:3, 3:6] (3x3) -> C[0:3, 3:6] (3x3)
    # 4. A[0:3, 3:5] (3x2) * B[3:5, 3:6] (2x3) -> C[0:3, 3:6] (3x3)
    # 5. A[3:4, 0:3] (1x3) * B[0:3, 0:3] (3x3) -> C[3:4, 0:3] (1x3)
    # 6. A[3:4, 3:5] (1x2) * B[3:5, 0:3] (2x3) -> C[3:4, 0:3] (1x3)
    # 7. A[3:4, 0:3] (1x3) * B[0:3, 3:6] (3x3) -> C[3:4, 3:6] (1x3)
    # 8. A[3:4, 3:5] (1x2) * B[3:5, 3:6] (2x3) -> C[3:4, 3:6] (1x3)
    
    # For each product, we need a tensor decomposition.
    # The global indices are:
    # U index: i*m + j where i is row of A (0-indexed), j is col of A
    # V index: j*p + k where j is row of B, k is col of B
    # W index: k*n + i where k is col of B, i is row of A (for C)
    
    # Let me build this step by step.
    
    # First, let me get the 3x3x3 Lader decomposition (rank 23)
    # This is complex. Let me use schoolbook for now and see if I can improve.
    
    # Actually, let me just use schoolbook for everything first to verify the framework,
    # then replace with better decompositions.
    
    # Schoolbook for a (a x b) * (b x c) product:
    # For each (i, j, k) in the sub-product, add a rank-1 term.
    
    def schoolbook_block(U, V, W, i0, j0, k0, ai, bj, ck, col_offset):
        """
        Add schoolbook terms for A[i0:i0+ai, j0:j0+bj] * B[j0:j0+bj, k0:k0+ck]
        contributing to C[i0:i0+ai, k0:k0+ck]
        """
        for ii in range(ai):
            for jj in range(bj):
                for kk in range(ck):
                    i = i0 + ii
                    j = j0 + jj
                    k = k0 + kk
                    U[i * m + j][col_offset] = 1
                    V[j * p + k][col_offset] = 1
                    W[k * n + i][col_offset] = 1
                    col_offset += 1
        return col_offset
    
    # Total schoolbook: 4*5*6 = 120
    # Let me just use schoolbook to verify, then optimize.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: A[0:3,0:3] * B[0:3,0:3] -> C[0:3,0:3]
    col = schoolbook_block(U, V, W, 0, 0, 0, 3, 3, 3, col)
    # Block 2: A[0:3,3:5] * B[3:5,0:3] -> C[0:3,0:3]
    col = schoolbook_block(U, V, W, 0, 3, 0, 3, 2, 3, col)
    # Block 3: A[0:3,0:3] * B[0:3,3:6] -> C[0:3,3:6]
    col = schoolbook_block(U, V, W, 0, 0, 3, 3, 3, 3, col)
    # Block 4: A[0:3,3:5] * B[3:5,3:6] -> C[0:3,3:6]
    col = schoolbook_block(U, V, W, 0, 3, 3, 3, 2, 3, col)
    # Block 5: A[3:4,0:3] * B[0:3,0:3] -> C[3:4,0:3]
    col = schoolbook_block(U, V, W, 3, 0, 0, 1, 3, 3, col)
    # Block 6: A[3:4,3:5] * B[3:5,0:3] -> C[3:4,0:3]
    col = schoolbook_block(U, V, W, 3, 3, 0, 1, 2, 3, col)
    # Block 7: A[3:4,0:3] * B[0:3,3:6] -> C[3:4,3:6]
    col = schoolbook_block(U, V, W, 3, 0, 3, 1, 3, 3, col)
    # Block 8: A[3:4,3:5] * B[3:5,3:6] -> C[3:4,3:6]
    col = schoolbook_block(U, V, W, 3, 3, 3, 1, 2, 3, col)
    
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}
