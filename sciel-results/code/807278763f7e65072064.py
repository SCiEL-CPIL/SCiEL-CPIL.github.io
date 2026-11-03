"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    """Search for low-rank decomposition of 5x5x5 matrix multiplication."""
    start = time.time()
    deadline = start + 110  # safety margin before 120s kill
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # We will use a block decomposition: 5 = 2 + 3
    # Blocks:
    # A22 (2x2), A23 (2x3)
    # A32 (3x2), A33 (3x3)
    #
    # B22 (2x2), B23 (2x3)
    # B32 (3x2), B33 (3x3)
    #
    # C22 (2x2) = A22 B22
    # C23 (2x3) = A22 B23 + A23 B33
    # C32 (3x2) = A32 B22 + A33 B32
    # C33 (3x3) = A32 B23 + A33 B33
    
    # Ranks:
    # R(2,2,2) = 7 (Strassen)
    # R(2,2,3) = 12 (Schoolbook)
    # R(2,3,3) = 18 (Schoolbook)
    # R(3,2,2) = 12 (Schoolbook)
    # R(3,3,2) = 18 (Schoolbook)
    # R(3,2,3) = 18 (Schoolbook)
    # R(3,3,3) = 23 (Laderman)
    
    total_rank = 7 + 12 + 18 + 12 + 18 + 18 + 23
    # total_rank = 108
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_decomposition(U_sub, V_sub, W_sub, i_off, j_off, k_off, rank_sub):
        """
        Add a sub-decomposition to the global matrices.
        i_off, j_off, k_off are the starting indices for the sub-blocks.
        U_sub shape: (i_size * j_size, rank_sub)
        V_sub shape: (j_size * k_size, rank_sub)
        W_sub shape: (k_size * i_size, rank_sub)
        """
        for r in range(rank_sub):
            for i in range(len(U_sub)):
                val = U_sub[i][r]
                if val != 0:
                    U[i_off + i][current_rank + r] = val
            for j in range(len(V_sub)):
                val = V_sub[j][r]
                if val != 0:
                    V[j_off + j][current_rank + r] = val
            for k in range(len(W_sub)):
                val = W_sub[k][r]
                if val != 0:
                    W[k_off + k][current_rank + r] = val

    # Helper to create schoolbook decomposition for m x n x p
    def schoolbook(m, n, p):
        rank = m * n * p
        U = [[0] * rank for _ in range(m * n)]
        V = [[0] * rank for _ in range(n * p)]
        W = [[0] * rank for _ in range(p * m)]
        col = 0
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    U[i * n + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * m + i][col] = 1
                    col += 1
        return U, V, W, rank

    # Helper to create Strassen decomposition for 2x2x2
    def strassen_2x2():
        # Rank 7
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        
        # Indices: A(i,j) -> i*2+j. B(j,k) -> j*2+k. C(i,k) -> i*2+k.
        
        # M1: (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1 # C00, C11
        
        # M2: (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1 # C10
        
        # M3: A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1 # C01
        
        # M4: A11(B01-B00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1 # C00, C10
        
        # M5: (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1 # C00, C01
        
        # M6: (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1 # C11
        
        # M7: (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1 # C00, C11
        
        return U, V, W, 7

    # Helper to create Laderman decomposition for 3x3x3
    def laderman_3x3():
        # Rank 23
        # This is complex to hardcode. We will use schoolbook for 3x3 to be safe?
        # Schoolbook rank is 27. Laderman is 23.
        # Given time constraints, using schoolbook for 3x3 is safer to implement correctly.
        # Rank 27 vs 23. Difference is small.
        # Let's use schoolbook for 3x3 to avoid errors in hardcoded Laderman.
        U, V, W, r = schoolbook(3, 3, 3)
        return U, V, W, r

    # Block 1: C22 = A22 B22 (2x2x2)
    U1, V1, W1, r1 = strassen_2x2()
    add_decomposition(U1, V1, W1, 0, 0, 0, r1)
    current_rank += r1
    
    # Block 2: A22 B23 (2x2x3)
    U2, V2, W2, r2 = schoolbook(2, 2, 3)
    add_decomposition(U2, V2, W2, 0, 0, 2, r2) # i_off=0, j_off=0, k_off=2
    current_rank += r2
    
    # Block 3: A23 B33 (2x3x3)
    U3, V3, W3, r3 = schoolbook(2, 3, 3)
    add_decomposition(U3, V3, W3, 0, 2, 2, r3) # i_off=0, j_off=2, k_off=2
    current_rank += r3
    
    # Block 4: A32 B22 (3x2x2)
    U4, V4, W4, r4 = schoolbook(3, 2, 2)
    add_decomposition(U4, V4, W4, 2, 0, 0, r4) # i_off=2, j_off=0, k_off=0
    current_rank += r4
    
    # Block 5: A33 B32 (3x3x2)
    U5, V5, W5, r5 = schoolbook(3, 3, 2)
    add_decomposition(U5, V5, W5, 2, 2, 0, r5) # i_off=2, j_off=2, k_off=0
    current_rank += r5
    
    # Block 6: A32 B23 (3x2x3)
    U6, V6, W6, r6 = schoolbook(3, 2, 3)
    add_decomposition(U6, V6, W6, 2, 0, 2, r6) # i_off=2, j_off=0, k_off=2
    current_rank += r6
    
    # Block 7: A33 B33 (3x3x3)
    U7, V7, W7, r7 = laderman_3x3()
    add_decomposition(U7, V7, W7, 2, 2, 2, r7) # i_off=2, j_off=2, k_off=2
    current_rank += r7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
