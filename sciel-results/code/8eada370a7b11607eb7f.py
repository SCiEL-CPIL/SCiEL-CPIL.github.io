import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    """
    Construct rank-42 Strassen-based decomposition for 2x2 * 2x12.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define Strassen vectors for one 2x2 block
    # Local indices for U (A): 00, 01, 10, 11
    # Local indices for V (B_q): 0,2q; 0,2q+1; 1,2q; 1,2q+1 -> 0, 1, 2, 3
    # Local indices for W (C_q): 2q,0; 2q,1; 2q+1,0; 2q+1,1 -> 0, 1, 2, 3
    
    U_local = [
        [1, 0, 0, 0], # P1: A00
        [1, 1, 0, 0], # P2: A00+A01
        [0, 0, 1, 0], # P3: A10
        [0, 0, 0, 1], # P4: A11
        [1, 0, 0, 1], # P5: A00+A11
        [0, 0, 1, 1], # P6: A10+A11
        [1, 1, 0, 0]  # P7: A00+A01
    ]
    
    V_local = [
        [1, 0, 0, 1], # P1: B00+B11
        [0, 0, 0, 1], # P2: B11
        [1, 1, 0, 0], # P3: B00+B01
        [-1, 0, 1, 0],# P4: B10-B00
        [1, 0, 0, 1], # P5: B00+B11
        [0, -1, 0, 1],# P6: B11-B01
        [1, 0, -1, 0] # P7: B00-B10
    ]
    
    W_local = [
        [0, 1, 0, 1], # P1: C01, C11
        [-1, 1, 0, 0],# P2: -C00, C01
        [0, 0, 1, 1], # P3: C10, C11
        [1, 0, 0, -1],# P4: C00, -C11
        [1, 0, 0, 0], # P5: C00
        [1, 0, 0, 0], # P6: C00
        [0, 0, 1, 1]  # P7: C10, C11
    ]
    
    # Fill U (same for all blocks)
    for r in range(7):
        U[0:4, r] = U_local[r]
        # Note: U is shared across all 6 blocks? 
        # No, U is (n*m, rank). Rank is 42.
        # The U part for block q is at columns 7*q to 7*q+6.
        # Since A is the same, the U vectors are the same for each block.
        pass

    # Actually, we need to fill U for each block's columns
    for q in range(6):
        for r in range(7):
            U[0:4, 7*q + r] = U_local[r]
            
            # Fill V
            # V indices: j*p + k
            # Local V indices 0,1,2,3 correspond to:
            # 0: j=0, k=2q -> idx 2q
            # 1: j=0, k=2q+1 -> idx 2q+1
            # 2: j=1, k=2q -> idx 12+2q
            # 3: j=1, k=2q+1 -> idx 12+2q+1
            
            V_idx = [2*q, 2*q+1, 12+2*q, 12+2*q+1]
            for i in range(4):
                V[V_idx[i], 7*q + r] = V_local[r][i]
                
            # Fill W
            # W indices: k*n + i
            # Local W indices 0,1,2,3 correspond to:
            # 0: k=2q, i=0 -> idx 4q
            # 1: k=2q, i=1 -> idx 4q+1
            # 2: k=2q+1, i=0 -> idx 4q+2
            # 3: k=2q+1, i=1 -> idx 4q+3
            
            W_idx = [4*q, 4*q+1, 4*q+2, 4*q+3]
            for i in range(4):
                W[W_idx[i], 7*q + r] = W_local[r][i]
                
    return {"U": U, "V": V, "W": W}
