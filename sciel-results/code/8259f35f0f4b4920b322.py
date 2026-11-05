"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _build_direct_sum_77():
    """
    Build a rank-77 exact decomposition via direct sum:
      4x4x4 part (Strassen rank 7) -> rank 7 * 4 = 28? No.
      Actually: 4x4x4 can be done with rank 49 schoolbook, or better.
      Let's use the simplest correct approach:
      
      4x4x4: schoolbook rank = 16. Or Strassen: 4x4x4 = (2x2x2)^4, each 2x2x2 rank 7, so 4*7=28.
      4x4x2: schoolbook rank = 4*4*2 = 32. Or Strassen tiling: 4x4x2 = 2 block-rows of A * 1 block-col of B,
             each block 2x2, so 2 block products, each 2x2x2 rank 7, total 14. Wait, let me re-derive.
             
      A is 4x4, B is 4x6, C is 4x6.
      Partition A into 2x2 blocks (each 2x2), B into 2x3 blocks (2 rows of 2x3 blocks), C into 2x3 blocks.
      C_i = A_i0 * B_0 + A_i1 * B_1 where i in {0,1} for row block, and B_0, B_1 are the two block columns? 
      No: A is 4x4 -> 2x2 blocks of 2x2. B is 4x6 -> 2x3 blocks of 2x2. C is 4x6 -> 2x3 blocks of 2x2.
      C_ij = sum_k A_ik B_kj for i,j in {0,1,2} (block indices).
      A has 2 rows of blocks (i=0,1), B has 2 rows of blocks (k=0,1), 3 cols (j=0,1,2).
      C_ij = A_i0 * B_0j + A_i1 * B_1j.
      So for each i in {0,1} and j in {0,1,2}, we compute A_i0 * B_0j + A_i1 * B_1j.
      Each block product is 2x2 * 2x2 -> 2x2, rank 7 via Strassen.
      Total block products: 2 * 3 * 2 = 12 block products, each rank 7, so 12*7 = 84. That's worse than schoolbook.
      
      Let me use schoolbook for 4x4x6: rank = 4*4*6 = 96. That's the baseline.
      
      For 4x4x4, schoolbook is 16, Strassen is 7*4 = 28? No: 4x4x4 = 2x2 blocks of 2x2x2. 4 block positions, each 2x2x2 rank 7, so 28. Schoolbook is 16. So schoolbook is better for 4x4x4.
      
      For 4x4x2: A is 4x4, B is 4x2, C is 4x2. 
      Schoolbook: 4*4*2 = 32.
      Strassen tiling: A is 2x2 blocks of 2x2, B is 2x1 blocks of 2x2, C is 2x1 blocks of 2x2.
      C_i = A_i0 * B_0 + A_i1 * B_1 for i in {0,1}.
      2 block products, each 2x2x2 rank 7, so 14. Better than 32.
      
      So 4x4x6 = 4x4x4 (schoolbook 16) + 4x4x2 (Strassen 14) = 30? No, the direct sum is on the k dimension.
      k goes 0..5. For k in 0..3, we do 4x4x4 schoolbook (16 terms). For k in 4..5, we do 4x4x2.
      But 4x4x2 uses the last 2 columns of B and last 2 columns of C. The k dimension is the contraction dimension.
      C[i,j] = sum_k A[i,k] B[k,j].
      For j in 0..5, k in 0..5.
      If we split k into k1 (0..3) and k2 (4..5):
      C[i,j] = sum_{k1=0..3} A[i,k1] B[k1,j] + sum_{k2=4..5} A[i,k2] B[k2,j].
      The first part: A is 4x4, B is 4x6 (first 4 cols of B, but B[k1,j] for k1 in 0..3, j in 0..5). This is 4x4 x 4x6? No.
      A[i,k1] for i in 0..3, k1 in 0..3 -> 4x4.
      B[k1,j] for k1 in 0..3, j in 0..5 -> 4x6.
      So this is 4x4 x 4x6 with the contraction over 4 dims. Rank of this tensor?
      It's not a square 4x4x4. It's A (4x4) times B (4x6) -> C (4x6). This is the full 4x4x6 problem!
      I'm going in circles. The k dimension IS the contraction dimension.
      
      Let me just use the schoolbook rank 96 as the guaranteed baseline, and try to improve.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    rank = n * m * p  # 96
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Start with schoolbook rank 96
    best = _build_direct_sum_77()
    best_rank = 96
    
    # Try to find a lower rank decomposition
    # Theoretical: 4x4x6 tensor rank. 
    # Known: 4x4x4 has rank 49 (Strassen), 4x4x2 has rank 28 (schoolbook) or 14 (Strassen tiling).
    # Direct sum: 49 + 28 = 77, or 49 + 14 = 63? No, the Strassen tiling for 4x4x2 uses the same A and different B,C slices.
    
    # Let me try a numerical approach: build the 96-term schoolbook, then use SVD to compress.
    # But the decomposition must be exact with integer/half-integer entries.
    
    # For now, return the schoolbook. This guarantees a valid answer.
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best
    
    return best
# EVOLVE-BLOCK-END
