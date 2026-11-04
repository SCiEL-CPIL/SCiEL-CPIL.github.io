"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    # Split B into B1 (m x 4) and B2 (m x 3)
    p1 = 4
    p2 = 3
    
    # Part 1: 3 x 4 x 4 using 4x4x4 schoolbook (rank 16)
    # Pad A to 4x4
    rank1 = 16
    U1 = np.zeros((4 * 4, rank1))
    V1 = np.zeros((4 * 4, rank1))
    W1 = np.zeros((4 * 4, rank1))
    
    c = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U1[i * 4 + j, c] = 1
                V1[j * 4 + k, c] = 1
                W1[k * 4 + i, c] = 1
                c += 1
                
    # Extract relevant rows for 3x4x4
    # U: i in 0..2, j in 0..3 -> rows 0..11
    U1_extracted = U1[0:12, :]
    # V: j in 0..3, k in 0..3 -> rows 0..15
    V1_extracted = V1
    # W: k in 0..3, i in 0..2 -> rows k*4+i where i in 0..2
    W1_rows = []
    for k in range(4):
        for i in range(3):
            W1_rows.append(k * 4 + i)
    W1_extracted = W1[W1_rows, :]
    
    # Part 2: 3 x 4 x 3 schoolbook (rank 36)
    rank2 = 36
    U2 = np.zeros((3 * 4, rank2))
    V2 = np.zeros((4 * 3, rank2))
    W2 = np.zeros((3 * 3, rank2))
    
    c = 0
    for i in range(3):
        for j in range(4):
            for k in range(3):
                U2[i * 4 + j, c] = 1
                V2[j * 3 + k, c] = 1
                W2[k * 3 + i, c] = 1
                c += 1
                
    # Combine
    total_rank = rank1 + rank2
    
    U = np.zeros((3 * 4, total_rank))
    V = np.zeros((4 * 7, total_rank))
    W = np.zeros((7 * 3, total_rank))
    
    # Fill U
    U[:, 0:rank1] = U1_extracted
    U[:, rank1:total_rank] = U2
    
    # Fill V
    # V1 corresponds to B1 (cols 0..3)
    # V indices j*7+k. For k=0..3
    for j in range(4):
        for k in range(4):
            V[j * 7 + k, 0:rank1] = V1_extracted[j * 4 + k, :]
            
    # V2 corresponds to B2 (cols 4..6)
    # V indices j*7+k. For k=4..6
    for j in range(4):
        for k in range(3):
            V[j * 7 + (4 + k), rank1:total_rank] = V2[j * 3 + k, :]
            
    # Fill W
    # W1 corresponds to C1 (cols 0..3)
    # W indices k*3+i. For k=0..3
    for k in range(4):
        for i in range(3):
            W[k * 3 + i, 0:rank1] = W1_extracted[k * 3 + i, :] # Wait, W1_extracted row index was k*4+i in original, but here we mapped it.
            # Actually W1_extracted is already ordered by k*3+i?
            # My W1_rows list was [0,1,2, 4,5,6, 8,9,10, 12,13,14].
            # These correspond to k=0,i=0..2; k=1,i=0..2; etc.
            # So W1_extracted[r, :] corresponds to W[k*3+i, :] where r is the index in the list.
            # Let's just use the row index directly.
            
    # Correct W filling
    # W1_extracted row r corresponds to global W row k*3+i
    # The order in W1_rows was:
    # k=0: 0,1,2 -> W rows 0,1,2
    # k=1: 4,5,6 -> W rows 3,4,5 ? No.
    # Global W index is k*3+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=0, i=2 -> 2
    # k=1, i=0 -> 3
    # k=1, i=1 -> 4
    # k=1, i=2 -> 5
    # So W1_extracted row 0 -> W row 0
    # W1_extracted row 1 -> W row 1
    # ...
    # W1_extracted row 3 -> W row 3
    # So we can just copy W1_extracted to W[0:12, 0:rank1]
    W[0:12, 0:rank1] = W1_extracted
    
    # W2 corresponds to C2 (cols 4..6)
    # Global W index k*3+i where k=4..6
    # k=4, i=0 -> 12
    # k=4, i=1 -> 13
    # k=4, i=2 -> 14
    # k=5, i=0 -> 15
    # ...
    # W2 rows are k_local*3+i where k_local=0..2
    # So W2[0:3] -> W[12:15]
    # W2[3:6] -> W[15:18]
    # W2[6:9] -> W[18:21]
    W[12:21, rank1:total_rank] = W2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
