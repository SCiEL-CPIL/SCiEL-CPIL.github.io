import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a rank-28 exact tensor decomposition for 2x2 * 2x8 matrix multiplication.
    Uses Strassen's algorithm for each of the 4 blocks of 2x2 in the second matrix.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 8
    rank = 28
    
    # Initialize matrices as zeros
    # U: (n*m, rank) -> (4, 28)
    # V: (m*p, rank) -> (16, 28)
    # W: (p*n, rank) -> (16, 28)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen terms for 2x2 * 2x2
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices (local): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Linear forms for A (rows of U)
    # M1: a+d -> [1, 0, 0, 1]
    # M2: c+d -> [0, 0, 1, 1]
    # M3: a   -> [1, 0, 0, 0]
    # M4: d   -> [0, 0, 0, 1]
    # M5: a+b -> [1, 1, 0, 0]
    # M6: c-a -> [-1, 0, 1, 0]
    # M7: b-d -> [0, 1, 0, -1]
    
    A_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # Linear forms for B (local block indices)
    # M1: x0+y1 -> [1, 0, 0, 1] (x0 is (0,0), y1 is (1,1))
    # M2: x0    -> [1, 0, 0, 0]
    # M3: x1-y1 -> [0, 1, 0, -1]
    # M4: y0-x0 -> [-1, 0, 1, 0]
    # M5: y1    -> [0, 0, 0, 1]
    # M6: x0+x1 -> [1, 1, 0, 0]
    # M7: y0+y1 -> [0, 0, 1, 1]
    
    B_forms = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Coefficients for C reconstruction
    # C00: M1 + M4 - M5 + M7 -> [1, 0, 0, 1, -1, 0, 1] (indices 0..6)
    # C01: M3 + M5           -> [0, 0, 1, 0, 1, 0, 0]
    # C10: M2 + M4           -> [0, 1, 0, 1, 0, 0, 0]
    # C11: M1 - M2 + M3 + M6 -> [1, -1, 1, 0, 0, 1, 0]
    
    C_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    for block in range(4):
        # Global column offset for this block's rank terms
        col_offset = block * 7
        
        # Fill U
        # U has shape (4, 28). Rows 0..3 correspond to A elements.
        # For each Strassen term s (0..6), we add the linear form to U[:, col_offset+s]
        for s in range(7):
            U[:, col_offset + s] = A_forms[s]
            
        # Fill V
        # V has shape (16, 28). Rows 0..15 correspond to B elements (m*p = 2*8).
        # B element (j, k) is at index j*p + k = j*8 + k.
        # For block `block`, the local k is 0..1. Global k = block*2 + k_local.
        for s in range(7):
            # B_forms[s] is [v00, v01, v10, v11] for local indices
            v00, v01, v10, v11 = B_forms[s]
            
            # Local index 0 -> (j=0, k_local=0) -> Global (0, block*2) -> Index 0*8 + block*2
            # Local index 1 -> (j=0, k_local=1) -> Global (0, block*2+1) -> Index 0*8 + block*2+1
            # Local index 2 -> (j=1, k_local=0) -> Global (1, block*2) -> Index 1*8 + block*2
            # Local index 3 -> (j=1, k_local=1) -> Global (1, block*2+1) -> Index 1*8 + block*2+1
            
            V[0 * p + block * 2, col_offset + s] = v00
            V[0 * p + block * 2 + 1, col_offset + s] = v01
            V[1 * p + block * 2, col_offset + s] = v10
            V[1 * p + block * 2 + 1, col_offset + s] = v11
            
        # Fill W
        # W has shape (16, 28). Rows 0..15 correspond to C elements (p*n = 8*2).
        # C element (k, i) is at index k*n + i = k*2 + i.
        # For block `block`, the local k is 0..1. Global k = block*2 + k_local.
        # C_coeffs[c_idx] gives coefficients for M0..M6.
        # c_idx 0 -> C00 (i=0, k_local=0)
        # c_idx 1 -> C01 (i=1, k_local=0) -- Wait, C index convention?
        # Task says: W[k*n+i, r].
        # C is p x n. p=8, n=2.
        # C[k, i]. k is 0..7, i is 0..1.
        # Block 0 corresponds to k=0,1.
        #   k=0: C[0,0] (i=0), C[0,1] (i=1)
        #   k=1: C[1,0] (i=0), C[1,1] (i=1)
        # So for block 0:
        #   C[0,0] is c_idx 0 in my local C_coeffs?
        #   My C_coeffs[0] is for C00 (top-left).
        #   My C_coeffs[1] is for C01 (top-right).
        #   My C_coeffs[2] is for C10 (bottom-left).
        #   My C_coeffs[3] is for C11 (bottom-right).
        
        # Mapping:
        # Local C00 -> Global C[0, 0] -> Index 0*2 + 0 = 0
        # Local C01 -> Global C[0, 1] -> Index 0*2 + 1 = 1
        # Local C10 -> Global C[1, 0] -> Index 1*2 + 0 = 2
        # Local C11 -> Global C[1, 1] -> Index 1*2 + 1 = 3
        
        # For general block:
        # Global k = block*2 + k_local
        # Global i = i_local (0 or 1)
        # Index = Global k * n + Global i = (block*2 + k_local)*2 + i_local
        
        for s in range(7):
            for c_idx in range(4):
                coeff = C_coeffs[c_idx][s]
                if coeff != 0:
                    # Determine local i and k_local from c_idx
                    # c_idx 0: i=0, k_loc=0
                    # c_idx 1: i=1, k_loc=0
                    # c_idx 2: i=0, k_loc=1
                    # c_idx 3: i=1, k_loc=1
                    i_local = c_idx % 2
                    k_local = c_idx // 2
                    
                    global_k = block * 2 + k_local
                    global_i = i_local
                    w_idx = global_k * n + global_i
                    
                    W[w_idx, col_offset + s] = coeff

    # Check deadline (should be well within)
    if time.time() > deadline:
        # This shouldn't happen, but if it does, return what we have
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
