"""
Rank-minimal tensor decomposition for 2x2 by 2x4 matrix multiplication.
Rank is 14.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Return the known optimal rank-14 decomposition immediately.
    # This avoids any search time and guarantees the minimum known rank.
    
    # Shapes:
    # U: (n*m, rank) = (4, 14)
    # V: (m*p, rank) = (8, 14)
    # W: (p*n, rank) = (8, 14)
    
    # Indexing conventions based on the problem statement:
    # A is 2x2. U index: i*m + j (i=0..1, j=0..1). 0:00, 1:01, 2:10, 3:11.
    # B is 2x4. V index: j*p + k (j=0..1, k=0..3). 0:00, 1:01, 2:02, 3:03, 4:10, 5:11, 6:12, 7:13.
    # C is 2x4. W index: k*n + i (k=0..3, i=0..1). 0:00, 1:10, 2:01, 3:11, 4:02, 5:12, 6:03, 7:13.
    # Note: The prompt says W[k*n+i, r].
    # C[i,k] corresponds to W[k*n + i].
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7
    
    rank = 14
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(8)]
    W = [[0] * rank for _ in range(8)]
    
    # Two independent Strassen blocks.
    # Block 1: C[:, 0:2] = A @ B[:, 0:2]
    # Block 2: C[:, 2:4] = A @ B[:, 2:4]
    
    # --- Strassen Block 1 (cols 0,1) ---
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # --- Strassen Block 2 (cols 2,3) ---
    # N1
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
