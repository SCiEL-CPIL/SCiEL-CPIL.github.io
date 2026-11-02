"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Rank-42 Strassen decomposition for 2x2 * 2x12 matrix multiplication.
    
    Partition 12 columns into 6 groups of 2. For each 2x2 * 2x2 block,
    use Strassen's rank-7 decomposition.
    Total rank = 6 * 7 = 42.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_blocks = p // 2  # 6
    rank = num_blocks * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strassen 7-product decomposition for 2x2x2
    # A: 0=a, 1=b, 2=c, 3=d
    # B: 0=e, 1=f, 2=g, 3=h
    # C: 0=C00, 1=C01, 2=C10, 3=C11
    
    # U coefficients (for A)
    # M1: a+d -> [1, 0, 0, 1]
    # M2: c+d -> [0, 0, 1, 1]
    # M3: a   -> [1, 0, 0, 0]
    # M4: d   -> [0, 0, 0, 1]
    # M5: a+b -> [1, 1, 0, 0]
    # M6: c-a -> [-1, 0, 1, 0]
    # M7: b-d -> [0, 1, 0, -1]
    
    U_block = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=np.float64).T # Shape (4, 7)
    
    # V coefficients (for B block)
    # M1: e+h -> [1, 0, 0, 1]
    # M2: e   -> [1, 0, 0, 0]
    # M3: f-h -> [0, 1, 0, -1]
    # M4: g-e -> [-1, 0, 1, 0]
    # M5: h   -> [0, 0, 0, 1]
    # M6: e+f -> [1, 1, 0, 0]
    # M7: g+h -> [0, 0, 1, 1]
    
    V_block = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=np.float64).T # Shape (4, 7)
    
    # W coefficients (for C block)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W_block[i, s] is coeff of M_s in C_i
    W_block = np.array([
        [1, 0, 0, 1, 0, 0, 0],   # C00: M1 + M4 - M5 + M7 -> wait.
        # C00 = M1 + M4 - M5 + M7
        # M1: 1, M2: 0, M3: 0, M4: 1, M5: -1, M6: 0, M7: 1
        [1, 0, 0, 1, -1, 0, 1],
        # C01 = M3 + M5
        # M1: 0, M2: 0, M3: 1, M4: 0, M5: 1, M6: 0, M7: 0
        [0, 0, 1, 0, 1, 0, 0],
        # C10 = M2 + M4
        # M1: 0, M2: 1, M3: 0, M4: 1, M5: 0, M6: 0, M7: 0
        [0, 1, 0, 1, 0, 0, 0],
        # C11 = M1 - M2 + M3 + M6
        # M1: 1, M2: -1, M3: 1, M4: 0, M5: 0, M6: 1, M7: 0
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=np.float64).T # Shape (4, 7)
    
    # Fill U
    # U is (4, 42). U_block is (4, 7).
    # Repeat U_block 6 times.
    U[:] = np.tile(U_block, (1, num_blocks))
    
    # Fill V
    # V is (24, 42).
    # For each block t (0..5):
    #   V[2t, col] = V_block[0, s]
    #   V[2t+1, col] = V_block[1, s]
    #   V[12+2t, col] = V_block[2, s]
    #   V[12+2t+1, col] = V_block[3, s]
    
    # We can construct V by iterating over blocks, or using advanced indexing.
    # Iteration is fast enough for 6 blocks.
    for t in range(num_blocks):
        if time.time() > deadline:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        cols = slice(t * 7, (t + 1) * 7)
        # B row 0, cols 2t, 2t+1
        V[2*t, cols] = V_block[0, :]
        V[2*t+1, cols] = V_block[1, :]
        # B row 1, cols 2t, 2t+1 -> global indices 12+2t, 12+2t+1
        V[12+2*t, cols] = V_block[2, :]
        V[12+2*t+1, cols] = V_block[3, :]
        
    # Fill W
    # W is (24, 42).
    # W index k*n+i.
    # For block t:
    #   C[0, 2t] -> W[4t, col]
    #   C[0, 2t+1] -> W[4t+2, col]
    #   C[1, 2t] -> W[4t+1, col]
    #   C[1, 2t+1] -> W[4t+3, col]
    
    for t in range(num_blocks):
        if time.time() > deadline:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        cols = slice(t * 7, (t + 1) * 7)
        W[4*t, cols] = W_block[0, :]
        W[4*t+2, cols] = W_block[1, :]
        W[4*t+1, cols] = W_block[2, :]
        W[4*t+3, cols] = W_block[3, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
