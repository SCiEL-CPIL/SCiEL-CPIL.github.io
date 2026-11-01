import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank based on Strassen decomposition of 2x2x2 blocks
    # 4 blocks of 2x2x2 (rank 7 each) + 1 block of 2x1x10 (rank 1)
    rank = 4 * 7 + 1
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # U coefficients (A factors)
    # Row: i*2 + j (local), mapped to global
    # Col: s (0..6)
    U_s = [
        [1, 0, 1, 0, 1, 0, 0], # i=0, j=0 (A00)
        [0, 0, 0, 0, 0, 0, 1], # i=0, j=1 (A01)
        [0, 1, 0, 0, 1, 0, 0], # i=1, j=0 (A10)
        [1, 1, 0, 1, 0, 1, -1] # i=1, j=1 (A11)
    ]
    
    # V coefficients (B factors)
    # Row: j*2 + k (local), mapped to global
    # Col: s (0..6)
    V_s = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0 (B00)
        [1, 0, -1, 1, 0, 0, 1], # j=0, k=1 (B01)
        [1, 0, 1, 0, 0, 1, 0], # j=1, k=0 (B10)
        [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1 (B11)
    ]
    
    # W coefficients (C combination)
    # Row: k*2 + i (local), mapped to global
    # Col: s (0..6)
    W_s = [
        [1, 0, 0, 1, -1, 0, -1], # k=0, i=0 (C00)
        [0, 0, 1, 0, 1, 0, 0],   # k=0, i=1 (C10) -- Wait, C10 is i=1, k=0?
        # Let's check indices.
        # C_ik. i is row of C, k is col of C.
        # Task W index: k*n + i.
        # So W[k*2 + i, s].
        # C00 (i=0, k=0): W[0, s] = 1, 0, 0, 1, -1, 0, -1
        # C01 (i=0, k=1): W[2, s] = 0, 0, 1, 0, 1, 0, 0
        # C10 (i=1, k=0): W[1, s] = 0, 1, 0, 1, 0, 0, 0
        # C11 (i=1, k=1): W[3, s] = 1, -1, 1, 0, 0, 0, 1
        
        # Let's re-verify Strassen C formulas:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # M1..M7 correspond to cols 0..6.
        # W[0, 0]=1, W[0, 3]=1, W[0, 4]=-1, W[0, 6]=-1
        # W[2, 2]=1, W[2, 4]=1
        # W[1, 1]=1, W[1, 3]=1
        # W[3, 0]=1, W[3, 1]=-1, W[3, 2]=1, W[3, 6]=1
        
        [0, 1, 0, 1, 0, 0, 0],   # k=0, i=1 (C10) -> Index 1
        [1, -1, 1, 0, 0, 0, 1]   # k=1, i=1 (C11) -> Index 3
    ]
    # Correct W_s order:
    # Index 0: k=0, i=0
    # Index 1: k=0, i=1
    # Index 2: k=1, i=0
    # Index 3: k=1, i=1
    W_s = [
        [1, 0, 0, 1, -1, 0, -1], # C00
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [1, -1, 1, 0, 0, 0, 1]   # C11
    ]

    col_offset = 0
    # 4 blocks of 2x2x2
    # j in {0,1}, k in {0,1}, {2,3}, {4,5}, {6,7}, {8,9}
    # Wait, I said 4 blocks.
    # Block 1: k=0,1
    # Block 2: k=2,3
    # Block 3: k=4,5
    # Block 4: k=6,7
    # Block 5: k=8,9
    # That's 5 blocks!
    # 10 / 2 = 5.
    # Rank = 5 * 7 + 1 = 36.
    
    for block_idx in range(5):
        k_start = block_idx * 2
        for s in range(7):
            r = col_offset + s
            # Fill U
            for i in range(2):
                for j in range(2):
                    u_idx = i * m + j
                    U[u_idx][r] = U_s[i*2 + j][s]
            # Fill V
            for j in range(2):
                for k in range(2):
                    v_idx = j * p + (k_start + k)
                    V[v_idx][r] = V_s[j*2 + k][s]
            # Fill W
            for k in range(2):
                for i in range(2):
                    w_idx = (k_start + k) * n + i
                    W[w_idx][r] = W_s[k*2 + i][s]
        col_offset += 7
        
    # Fill the j=2 part (Rank 1)
    r = col_offset
    for i in range(2):
        U[i * m + 2][r] = 1
    for k in range(10):
        V[2 * p + k][r] = 1
    for k in range(10):
        for i in range(2):
            W[k * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
