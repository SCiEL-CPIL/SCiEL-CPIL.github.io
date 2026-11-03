import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen for 4x4x4 blocks
    rank = 56
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # Returns indices for U, V, W for a 2x2x2 multiplication
    # Local indices: i, j, k in 0..1
    # Global indices for A: i*2 + j (if block is at 0,0)
    # We need to map local block indices to global indices.
    
    # Helper to add a Strassen decomposition for a specific 2x2 block multiplication
    # block_i, block_j: block row/col in the 4x4 matrix (0 or 1)
    # block_k: block col in the B matrix (0 or 1)
    # offset: column offset in U, V, W
    
    def add_strassen_2x2(block_i, block_j, block_k, offset):
        # Strassen terms for 2x2
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B12 - B22)
        # M4 = A22 (B21 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # Coefficients for A (U)
        # U indices: (block_i*2 + li, block_j*2 + lj)
        # V indices: (block_j*2 + lj, block_k*2 + lk)
        # W indices: (block_k*2 + lk, block_i*2 + li)
        
        # Term 1
        U[(block_i*2+0)*m + (block_j*2+0)][offset+0] += 1
        U[(block_i*2+1)*m + (block_j*2+1)][offset+0] += 1
        V[(block_j*2+0)*p + (block_k*2+0)][offset+0] += 1
        V[(block_j*2+1)*p + (block_k*2+1)][offset+0] += 1
        W[(block_k*2+0)*n + (block_i*2+0)][offset+0] += 1
        W[(block_k*2+1)*n + (block_i*2+1)][offset+0] += 1
        W[(block_k*2+0)*n + (block_i*2+1)][offset+0] -= 1
        W[(block_k*2+1)*n + (block_i*2+0)][offset+0] -= 1
        
        # Term 2
        U[(block_i*2+1)*m + (block_j*2+0)][offset+1] += 1
        U[(block_i*2+1)*m + (block_j*2+1)][offset+1] += 1
        V[(block_j*2+0)*p + (block_k*2+0)][offset+1] += 1
        W[(block_k*2+0)*n + (block_i*2+1)][offset+1] += 1
        W[(block_k*2+1)*n + (block_i*2+1)][offset+1] += 1
        W[(block_k*2+0)*n + (block_i*2+0)][offset+1] -= 1
        W[(block_k*2+1)*n + (block_i*2+0)][offset+1] -= 1
        
        # Term 3
        U[(block_i*2+0)*m + (block_j*2+0)][offset+2] += 1
        V[(block_j*2+0)*p + (block_k*2+1)][offset+2] += 1
        V[(block_j*2+1)*p + (block_k*2+1)][offset+2] -= 1
        W[(block_k*2+1)*n + (block_i*2+0)][offset+2] += 1
        W[(block_k*2+0)*n + (block_i*2+0)][offset+2] -= 1
        W[(block_k*2+1)*n + (block_i*2+1)][offset+2] -= 1
        W[(block_k*2+0)*n + (block_i*2+1)][offset+2] += 1
        
        # Term 4
        U[(block_i*2+1)*m + (block_j*2+1)][offset+3] += 1
        V[(block_j*2+1)*p + (block_k*2+0)][offset+3] += 1
        V[(block_j*2+0)*p + (block_k*2+0)][offset+3] -= 1
        W[(block_k*2+0)*n + (block_i*2+1)][offset+3] += 1
        W[(block_k*2+1)*n + (block_i*2+1)][offset+3] -= 1
        W[(block_k*2+0)*n + (block_i*2+0)][offset+3] -= 1
        W[(block_k*2+1)*n + (block_i*2+0)][offset+3] += 1
        
        # Term 5
        U[(block_i*2+0)*m + (block_j*2+0)][offset+4] += 1
        U[(block_i*2+0)*m + (block_j*2+1)][offset+4] += 1
        V[(block_j*2+1)*p + (block_k*2+1)][offset+4] += 1
        W[(block_k*2+1)*n + (block_i*2+0)][offset+4] += 1
        W[(block_k*2+1)*n + (block_i*2+1)][offset+4] += 1
        W[(block_k*2+0)*n + (block_i*2+0)][offset+4] -= 1
        W[(block_k*2+0)*n + (block_i*2+1)][offset+4] -= 1
        
        # Term 6
        U[(block_i*2+1)*m + (block_j*2+0)][offset+5] += 1
        U[(block_i*2+0)*m + (block_j*2+0)][offset+5] -= 1
        V[(block_j*2+0)*p + (block_k*2+0)][offset+5] += 1
        V[(block_j*2+0)*p + (block_k*2+1)][offset+5] += 1
        W[(block_k*2+0)*n + (block_i*2+1)][offset+5] += 1
        W[(block_k*2+1)*n + (block_i*2+1)][offset+5] += 1
        W[(block_k*2+0)*n + (block_i*2+0)][offset+5] -= 1
        W[(block_k*2+1)*n + (block_i*2+0)][offset+5] -= 1
        
        # Term 7
        U[(block_i*2+0)*m + (block_j*2+1)][offset+6] += 1
        U[(block_i*2+1)*m + (block_j*2+1)][offset+6] -= 1
        V[(block_j*2+1)*p + (block_k*2+0)][offset+6] += 1
        V[(block_j*2+1)*p + (block_k*2+1)][offset+6] += 1
        W[(block_k*2+0)*n + (block_i*2+0)][offset+6] += 1
        W[(block_k*2+1)*n + (block_i*2+0)][offset+6] += 1
        W[(block_k*2+0)*n + (block_i*2+1)][offset+6] -= 1
        W[(block_k*2+1)*n + (block_i*2+1)][offset+6] -= 1

    # We have 4 blocks of A (2x2) and 8 blocks of B (2x2) if we view 4x8 as 2x4 blocks of 2x2?
    # No, B is 4x8. Split into two 4x4 matrices B1, B2.
    # B1 has blocks (0,0), (0,1), (1,0), (1,1) in 2x2 grid.
    # B2 has blocks (0,2), (0,3), (1,2), (1,3) in 2x2 grid?
    # Let's index B blocks by (row_block, col_block) where col_block is 0..3.
    # B1: col_block 0, 1. B2: col_block 2, 3.
    
    offset = 0
    for block_i in range(2):
        for block_j in range(2):
            # B1
            add_strassen_2x2(block_i, block_j, 0, offset)
            offset += 7
            add_strassen_2x2(block_i, block_j, 1, offset)
            offset += 7
            # B2
            add_strassen_2x2(block_i, block_j, 2, offset)
            offset += 7
            add_strassen_2x2(block_i, block_j, 3, offset)
            offset += 7
            
    return {"U": U, "V": V, "W": W}
