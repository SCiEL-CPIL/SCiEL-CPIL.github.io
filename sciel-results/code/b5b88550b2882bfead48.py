"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Strassen-based rank-42 decomposition for 2x2 * 2x12 matrix multiplication.
    
    Partition 12 columns into 6 pairs of 2. Apply 7-product Strassen to each 2x2 block.
    Total rank = 6 * 7 = 42.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42
    
    # Initialize U, V, W
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A indices: 0:a, 1:b, 2:c, 3:d
    # B indices: 0:e, 1:f, 2:g, 3:h
    # C indices: 0:C00, 1:C01, 2:C10, 3:C11
    
    # U_block: 4x7
    U_block = [
        [1, 0, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ]
    
    # V_block: 4x7 (relative to block columns 0,1)
    V_block = [
        [1, 1, 0, -1, 0, 1, 0], # e (B[0][0])
        [1, 0, 1, 0, 0, 1, 0],  # f (B[0][1])
        [0, 0, 0, 1, 0, 0, 1],  # g (B[1][0])
        [1, 0, -1, 0, 1, 0, 1]  # h (B[1][1])
    ]
    
    # W_block: 4x7 (relative to block columns 0,1)
    W_block = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    for t in range(num_pairs):
        if time.time() > deadline:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        base_col = 2 * t
        rank_offset = t * 7
        
        # Fill U (same for all blocks)
        U[:, rank_offset:rank_offset+7] = U_block
        
        # Fill V
        # B rows: j*p + k
        # Block t uses B columns base_col and base_col+1
        # V_block row 0 corresponds to B[0][base_col] -> row 0*p + base_col = base_col
        # V_block row 1 corresponds to B[0][base_col+1] -> row 0*p + base_col+1 = base_col+1
        # V_block row 2 corresponds to B[1][base_col] -> row 1*p + base_col = p + base_col
        # V_block row 3 corresponds to B[1][base_col+1] -> row 1*p + base_col+1 = p + base_col+1
        
        v_rows = [base_col, base_col+1, p+base_col, p+base_col+1]
        for r in range(4):
            V[v_rows[r], rank_offset:rank_offset+7] = V_block[r]
            
        # Fill W
        # W rows: k*n + i
        # Block t uses C columns base_col and base_col+1
        # W_block row 0 corresponds to C[0][base_col] -> row base_col*n + 0
        # W_block row 1 corresponds to C[0][base_col+1] -> row (base_col+1)*n + 0
        # W_block row 2 corresponds to C[1][base_col] -> row base_col*n + 1
        # W_block row 3 corresponds to C[1][base_col+1] -> row (base_col+1)*n + 1
        
        w_rows = [
            base_col * n + 0,
            (base_col + 1) * n + 0,
            base_col * n + 1,
            (base_col + 1) * n + 1
        ]
        for r in range(4):
            W[w_rows[r], rank_offset:rank_offset+7] = W_block[r]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
