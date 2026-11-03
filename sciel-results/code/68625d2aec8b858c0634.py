"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen for 2x2 blocks
    # A is 2x4, B is 4x4, C is 2x4
    # Split A into A1 (2x2), A2 (2x2)
    # Split B into B1 (2x2), B2 (2x2), B3 (2x2), B4 (2x2)
    # C1 (2x2) = A1 B1 + A2 B3
    # C2 (2x2) = A1 B2 + A2 B4
    
    # Strassen rank-1 tensors for 2x2x2
    # U, V, W for one 2x2x2 multiplication
    # Indices in 2x2x2:
    # A: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    # B: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    # C: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    
    # Strassen factors (u, v, w) for the 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U factors (for A)
    U_strassen = [
        [1, 0, 1, 0], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # V factors (for B)
    V_strassen = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # W factors (for C)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M2
    W_strassen = [
        [1, 0, 0, -1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [1, 0, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [-1, 1, 0, 0]  # M7
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)] # 8 rows
    V = [[0] * rank for _ in range(m * p)] # 16 rows
    W = [[0] * rank for _ in range(p * n)] # 8 rows
    
    # We have 4 block multiplications:
    # 1. A1 B1 -> C1 part 1
    # 2. A1 B2 -> C2 part 1
    # 3. A2 B3 -> C1 part 2
    # 4. A2 B4 -> C2 part 2
    
    # Mapping of indices:
    # A: 2x4. A1 is cols 0,1. A2 is cols 2,3.
    # A indices:
    # A1: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # A2: (0,0)->4, (0,1)->5, (1,0)->6, (1,1)->7
    
    # B: 4x4.
    # B1: rows 0,1 cols 0,1. Indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B2: rows 0,1 cols 2,3. Indices: (0,0)->4, (0,1)->5, (1,0)->6, (1,1)->7
    # B3: rows 2,3 cols 0,1. Indices: (0,0)->8, (0,1)->9, (1,0)->10, (1,1)->11
    # B4: rows 2,3 cols 2,3. Indices: (0,0)->12, (0,1)->13, (1,0)->14, (1,1)->15
    
    # C: 2x4.
    # C1: cols 0,1. Indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C2: cols 2,3. Indices: (0,0)->4, (0,1)->5, (1,0)->6, (1,1)->7
    
    def add_block_mult(a_offset, b_offset, c_offset, target_rank_start):
        for k in range(7):
            col = target_rank_start + k
            # U
            for i in range(4):
                U[a_offset + i][col] += U_strassen[k][i]
            # V
            for i in range(4):
                V[b_offset + i][col] += V_strassen[k][i]
            # W
            for i in range(4):
                W[c_offset + i][col] += W_strassen[k][i]
        
    # Block 1: A1 (offset 0) * B1 (offset 0) -> C1 (offset 0)
    add_block_mult(0, 0, 0, 0)
    
    # Block 2: A1 (offset 0) * B2 (offset 4) -> C2 (offset 4)
    add_block_mult(0, 4, 4, 7)
    
    # Block 3: A2 (offset 4) * B3 (offset 8) -> C1 (offset 0)
    add_block_mult(4, 8, 0, 14)
    
    # Block 4: A2 (offset 4) * B4 (offset 12) -> C2 (offset 4)
    add_block_mult(4, 12, 4, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
