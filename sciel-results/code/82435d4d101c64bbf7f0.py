import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # Strassen components for 2x2 matrix multiplication
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # r=0 (M1): M1 = (A00+A11)(B00+B11)
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1
    W_local[3, 0] = 1
    
    # r=1 (M2): M2 = (A10+A11)B00
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1
    W_local[3, 1] = -1
    
    # r=2 (M3): M3 = A00(B10-B00)
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1
    W_local[3, 2] = 1
    
    # r=3 (M4): M4 = A11(B11-B01)
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1
    W_local[2, 3] = 1
    
    # r=4 (M5): M5 = (A00+A01)B11
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1
    W_local[1, 4] = 1
    
    # r=5 (M6): M6 = (A10-A00)(B00+B01)
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1
    
    # r=6 (M7): M7 = (A01-A11)(B10+B11)
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # We have 2 splits of the middle dimension (s=0,1) and 4 column pairs (j=0..3)
    # Total 8 blocks, each rank 7.
    
    for s in range(2):
        if s == 0:
            # A uses columns 0, 1
            # B uses rows 0, 1
            a_cols = [0, 1]
            b_rows = [0, 1]
        else:
            # A uses columns 2, 3
            # B uses rows 2, 3
            a_cols = [2, 3]
            b_rows = [2, 3]
            
        for j in range(4):
            base_r = (j * 2 + s) * 7
            
            # Map local indices (0,1,2,3) to global indices
            # Local 0: (0, a_cols[0]) -> A[0, a_cols[0]]
            # Local 1: (0, a_cols[1]) -> A[0, a_cols[1]]
            # Local 2: (1, a_cols[0]) -> A[1, a_cols[0]]
            # Local 3: (1, a_cols[1]) -> A[1, a_cols[1]]
            a_map = [
                0 * m + a_cols[0],
                0 * m + a_cols[1],
                1 * m + a_cols[0],
                1 * m + a_cols[1]
            ]
            
            # B uses rows b_rows[0], b_rows[1] and columns 2*j, 2*j+1
            # Local 0: (b_rows[0], 2*j)
            # Local 1: (b_rows[0], 2*j+1)
            # Local 2: (b_rows[1], 2*j)
            # Local 3: (b_rows[1], 2*j+1)
            b_map = [
                b_rows[0] * p + (2 * j),
                b_rows[0] * p + (2 * j + 1),
                b_rows[1] * p + (2 * j),
                b_rows[1] * p + (2 * j + 1)
            ]
            
            # W indices: k*n + i
            # Local 0: (i=0, k=2*j) -> 2*j*n + 0
            # Local 1: (i=0, k=2*j+1) -> (2*j+1)*n + 0
            # Local 2: (i=1, k=2*j) -> 2*j*n + 1
            # Local 3: (i=1, k=2*j+1) -> (2*j+1)*n + 1
            c_map = [
                (2 * j) * n + 0,
                (2 * j + 1) * n + 0,
                (2 * j) * n + 1,
                (2 * j + 1) * n + 1
            ]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
