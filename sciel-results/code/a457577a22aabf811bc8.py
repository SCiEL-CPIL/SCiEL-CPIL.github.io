import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 using Strassen decomposition for 6 blocks of 2x2x2
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen vectors for 2x2 matrix multiplication
    # A flat: [A00, A01, A10, A11]
    # B flat: [B00, B01, B10, B11]
    # C flat: [C00, C10, C01, C11] (col major)
    
    u_vectors = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    v_vectors = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1]   # M7: B10 + B11
    ]
    
    w_vectors = [
        [1, 0, 0, 1],  # M1: C00 + C11
        [0, 1, 0, -1], # M2: C10 - C11
        [0, 0, 1, 1],  # M3: C01 + C11
        [1, 1, 0, 0],  # M4: C00 + C10
        [-1, 0, 1, 0], # M5: -C00 + C01
        [0, 0, 0, 1],  # M6: C11
        [1, 0, 0, 0]   # M7: C00
    ]
    
    num_blocks = p // 2 # 6 blocks
    
    for g in range(num_blocks):
        for s in range(7):
            col_idx = g * 7 + s
            
            # U
            U[:, col_idx] = u_vectors[s]
            
            # V indices for block g
            # B is 2x12. Flat index r*p + c.
            # Block g uses cols 2g, 2g+1.
            b_00_idx = 0 * p + (2 * g)
            b_01_idx = 0 * p + (2 * g + 1)
            b_10_idx = 1 * p + (2 * g)
            b_11_idx = 1 * p + (2 * g + 1)
            
            V[b_00_idx, col_idx] = v_vectors[s][0]
            V[b_01_idx, col_idx] = v_vectors[s][1]
            V[b_10_idx, col_idx] = v_vectors[s][2]
            V[b_11_idx, col_idx] = v_vectors[s][3]
            
            # W indices for block g
            # W is p*n. Flat index k*n + i.
            # Block g uses cols 2g, 2g+1.
            w_00_idx = (2 * g) * n + 0
            w_10_idx = (2 * g) * n + 1
            w_01_idx = (2 * g + 1) * n + 0
            w_11_idx = (2 * g + 1) * n + 1
            
            W[w_00_idx, col_idx] = w_vectors[s][0]
            W[w_10_idx, col_idx] = w_vectors[s][1]
            W[w_01_idx, col_idx] = w_vectors[s][2]
            W[w_11_idx, col_idx] = w_vectors[s][3]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
