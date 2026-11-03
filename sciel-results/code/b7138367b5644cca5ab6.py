"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Constructive decomposition for 3x5x5
    # Split A (3x5) into A1 (3x4) and A2 (3x1)
    # Split B (5x5) into B11 (4x4), B12 (4x1), B21 (1x4), B22 (1x1)
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    
    # Part 1: A1*B11 (3x4x4). Use schoolbook rank 16? No, 4x4x4 is 16? No, 16 is 4^2.
    # Wait, schoolbook for 4x4x4 is 16 multiplications.
    # Rank of 4x4x4 is 16.
    # So Part 1 rank is 16.
    
    # Part 2: A1*B12 (3x4x1). Rank 12.
    # Part 3: A2*B21 (3x1x4). Rank 12.
    # Part 4: A2*B22 (3x1x1). Rank 3.
    
    # Total Rank = 16 + 12 + 12 + 3 = 43.
    
    rank_total = 75
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col_offset = 0
    
    # Part 1: A1 (3x4) * B11 (4x4) -> C1 (3x4)
    # Indices:
    # A1: i in 0..2, j in 0..3. Index in A: i*5 + j
    # B11: j in 0..3, k in 0..3. Index in B: j*5 + k
    # C1: i in 0..2, k in 0..3. Index in C: k*3 + i
    
    for i in range(3):
        for j in range(4):
            for k in range(4):
                U[i * 5 + j, col_offset] = 1
                V[j * 5 + k, col_offset] = 1
                W[k * 3 + i, col_offset] = 1
                col_offset += 1
                
    # Part 2: A1 (3x4) * B12 (4x1) -> C2 (3x1) at col 4
    # Indices:
    # A1: i in 0..2, j in 0..3. Index in A: i*5 + j
    # B12: j in 0..3, k=4. Index in B: j*5 + 4
    # C2: i in 0..2, k=4. Index in C: 4*3 + i = 12 + i
    
    for i in range(3):
        for j in range(4):
            U[i * 5 + j, col_offset] = 1
            V[j * 5 + 4, col_offset] = 1
            W[12 + i, col_offset] = 1
            col_offset += 1
            
    # Part 3: A2 (3x1) * B21 (1x4) -> C3 (3x4)
    # Indices:
    # A2: i in 0..2, j=4. Index in A: i*5 + 4
    # B21: j=4, k in 0..3. Index in B: 4*5 + k = 20 + k
    # C3: i in 0..2, k in 0..3. Index in C: k*3 + i
    
    for i in range(3):
        for k in range(4):
            U[i * 5 + 4, col_offset] = 1
            V[20 + k, col_offset] = 1
            W[k * 3 + i, col_offset] = 1
            col_offset += 1
            
    # Part 4: A2 (3x1) * B22 (1x1) -> C4 (3x1) at col 4
    # Indices:
    # A2: i in 0..2, j=4. Index in A: i*5 + 4
    # B22: j=4, k=4. Index in B: 4*5 + 4 = 24
    # C4: i in 0..2, k=4. Index in C: 12 + i
    
    for i in range(3):
        U[i * 5 + 4, col_offset] = 1
        V[24, col_offset] = 1
        W[12 + i, col_offset] = 1
        col_offset += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
