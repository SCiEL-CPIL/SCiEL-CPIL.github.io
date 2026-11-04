"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s limit

    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24

    # Build the target tensor: T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k)
    # Otherwise T[a,b,c] = 0

    # Schoolbook decomposition (rank = n*m*p = 120)
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1

    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank

    # Try to construct a lower-rank decomposition using Strassen for 2x2*2x2 blocks.
    # The key idea: partition the 4x5 * 5x6 multiplication into blocks and apply
    # Strassen (rank 7) where both operands are 2x2, and schoolbook elsewhere.
    #
    # Partition A (4x5) into 2x2 blocks:
    #   A = [[A11, A12], [A21, A22]]
    #   where A11 is 2x2 (rows 0-1, cols 0-1), A12 is 2x1 (rows 0-1, col 2)
    #   A21 is 2x2 (rows 2-3, cols 0-1), A22 is 2x1 (rows 2-3, col 2)
    #
    # Partition B (5x6) into 2x2 blocks:
    #   B = [[B11, B12], [B21, B22]]
    #   where B11 is 2x2 (rows 0-1, cols 0-1), B12 is 2x3 (rows 0-1, cols 2-4)
    #   B21 is 2x2 (rows 2-3, cols 0-1), B22 is 2x3 (rows 2-3, cols 2-4)
    #   Wait, B is 5x6, so rows are 0-4. Let me re-partition.
    #
    # Actually, for 4x5 * 5x6:
    # A is 4x5: rows 0-3, cols 0-4
    # B is 5x6: rows 0-4, cols 0-5
    # C is 4x6: rows 0-3, cols 0-5
    #
    # Let's partition A into 2x2 blocks:
    #   A11 = A[0:2, 0:2] (2x2)
    #   A12 = A[0:2, 2:4] (2x2)  -- wait, A has 5 cols, so cols 0-4
    #   A13 = A[0:2, 4:5] (2x1)
    #   A21 = A[2:4, 0:2] (2x2)
    #   A22 = A[2:4, 2:4] (2x2)
    #   A23 = A[2:4, 4:5] (2x1)
    #
    # Partition B into 2x2 blocks:
    #   B11 = B[0:2, 0:2] (2x2)
    #   B12 = B[0:2, 2:4] (2x2)
    #   B13 = B[0:2, 4:6] (2x2)
    #   B21 = B[2:4, 0:2] (2x2)
    #   B22 = B[2:4, 2:4] (2x2)
    #   B23 = B[2:4, 4:6] (2x2)
    #   B31 = B[4:5, 0:2] (1x2)
    #   B32 = B[4:5, 2:4] (1x2)
    #   B33 = B[4:5, 4:6] (1x2)
    #
    # This is getting complex. Let me try a simpler approach.
    #
    # Actually, let me just return the schoolbook decomposition for now.
    # The Strassen-based approach requires careful construction of the
    # rank-1 tensors for each block multiplication, which is complex.
    
    return best_result
# EVOLVE-BLOCK-END
