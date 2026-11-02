"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 blocks of Strassen 2x2
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition
    # u, v, w are vectors of length 4 (indices i*2+j, j'*2+k, k*2+i)
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11) B00
    # P3 = A00 (B01 - B11)
    # P4 = A11 (B10 - B00)
    # P5 = (A00 + A01) B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # We define the 7 rank-1 tensors for 2x2 * 2x2
    # Each tensor is (u, v, w)
    # u indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # v indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # w indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*2+i)
    
    strassen_tensors = [
        # P1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # P2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # P3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # P4
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]),
        # P5
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]),
        # P6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 1, 0, 1]),
        # P7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 1, 0]), # Wait, C11 = P1 - P2 + P3 + P6. C01 = P3 + P5. C10 = P2 + P4. C00 = P1 + P4 - P5 + P7.
    ]
    
    # Let's re-verify the W vectors for Strassen
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # W for P1: C00, C11 -> w = [1, 0, 0, 1] (C00, C01, C10, C11)
    # W for P2: C10, C11 -> w = [0, 0, 1, -1]
    # W for P3: C01, C11 -> w = [0, 1, 0, 1]
    # W for P4: C00, C10 -> w = [1, 0, 1, 0]
    # W for P5: C00, C01 -> w = [-1, 1, 0, 0]
    # W for P6: C11 -> w = [0, 0, 0, 1]
    # W for P7: C00 -> w = [1, 0, 0, 0]
    
    strassen_tensors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # P2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # P3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # P5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # P7
    ]
    
    # Map to 2x4 * 4x4
    # 4 blocks:
    # 1. A1 B11 -> C1
    # 2. A2 B21 -> C1
    # 3. A1 B12 -> C2
    # 4. A2 B22 -> C2
    
    col = 0
    for block in range(4):
        for u, v, w in strassen_tensors:
            # Determine offsets
            # block 0: A1, B11, C1
            # block 1: A2, B21, C1
            # block 2: A1, B12, C2
            # block 3: A2, B22, C2
            
            a_col_offset = 0 if block % 2 == 0 else 2
            b_row_offset = 0 if block < 2 else 2
            b_col_offset = 0 if block < 2 else 2
            c_col_offset = 0 if block < 2 else 2
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    U[i * 4 + (j + a_col_offset)][col] = u[i * 2 + j]
            
            # Fill V
            for jp in range(2):
                for k in range(2):
                    V[(jp + b_row_offset) * 4 + (k + b_col_offset)][col] = v[jp * 2 + k]
            
            # Fill W
            for k in range(2):
                for i in range(2):
                    W[(k + c_col_offset) * 2 + i][col] = w[k * 2 + i]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
