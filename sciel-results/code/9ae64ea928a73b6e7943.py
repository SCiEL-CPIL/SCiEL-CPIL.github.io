"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy: 
    Split B (3x8) into four 3x2 blocks.
    For each block, use a rank-11 decomposition for 2x3 * 3x2.
    The rank-11 decomposition combines Strassen (rank 7) for the 2x2 * 2x2 part 
    and standard outer product (rank 4) for the remaining 2x1 * 1x2 part.
    Total rank = 4 * 11 = 44.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Define the rank-11 decomposition for a single 2x3 * 3x2 block
    U_block, V_block, W_block = _rank11_2x3x2_decomposition()
    
    rank_per_block = 11
    num_blocks = p // 2  # 8 // 2 = 4
    total_rank = rank_per_block * num_blocks  # 44
    
    # Initialize full matrices
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 44
    
    # Fill in the blocks
    for t in range(num_blocks):
        col_start = t * rank_per_block
        
        # U is the same for all blocks (A is shared)
        for i in range(n * m):
            for c in range(rank_per_block):
                U[i][col_start + c] = U_block[i][c]
        
        # V mapping for block t
        # Local b in 0..5 maps to global v
        # b=0 (j=0,k=0) -> v=2t
        # b=1 (j=0,k=1) -> v=2t+1
        # b=2 (j=1,k=0) -> v=8+2t
        # b=3 (j=1,k=1) -> v=8+2t+1
        # b=4 (j=2,k=0) -> v=16+2t
        # b=5 (j=2,k=1) -> v=16+2t+1
        v_map = [2*t, 2*t+1, 8+2*t, 8+2*t+1, 16+2*t, 16+2*t+1]
        
        for b in range(6):
            global_v = v_map[b]
            for c in range(rank_per_block):
                V[global_v][col_start + c] = V_block[b][c]
        
        # W mapping for block t
        # Local d in 0..3 maps to global w
        # d=0 (k=0,i=0) -> w=4t
        # d=1 (k=0,i=1) -> w=4t+1
        # d=2 (k=1,i=0) -> w=4t+2
        # d=3 (k=1,i=1) -> w=4t+3
        w_map = [4*t, 4*t+1, 4*t+2, 4*t+3]
        
        for d in range(4):
            global_w = w_map[d]
            for c in range(rank_per_block):
                W[global_w][col_start + c] = W_block[d][c]
                
    return {"U": U, "V": V, "W": W}

def _rank11_2x3x2_decomposition():
    """
    Returns U, V, W for rank-11 decomposition of 2x3 * 3x2.
    U: (6, 11), V: (6, 11), W: (4, 11)
    """
    rank = 11
    U = [[0] * rank for _ in range(6)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(4)]
    
    # --- Strassen Part (Cols 0-6) ---
    # A_L (2x2) indices: 0,1,3,4
    # B_T (2x2) indices: 0,1,2,3
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10 + A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10 - B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01 - B00)
    U[4][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[4][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    # --- Outer Product Part (Cols 7-10) ---
    # A_R (2x1) indices: 2, 5
    # B_B (1x2) indices: 4, 5
    
    # P1 = A02 * B20 -> C00
    U[2][7] = 1
    V[4][7] = 1
    W[0][7] = 1
    
    # P2 = A02 * B21 -> C01
    U[2][8] = 1
    V[5][8] = 1
    W[1][8] = 1
    
    # P3 = A12 * B20 -> C10
    U[5][9] = 1
    V[4][9] = 1
    W[2][9] = 1
    
    # P4 = A12 * B21 -> C11
    U[5][10] = 1
    V[5][10] = 1
    W[3][10] = 1
    
    return U, V, W
# EVOLVE-BLOCK-END
