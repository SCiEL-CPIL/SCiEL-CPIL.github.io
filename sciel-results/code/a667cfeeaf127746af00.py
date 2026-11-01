"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 28 decomposition using Strassen's algorithm for 2x2x2 blocks
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 factors
    # U: 4x7, V: 4x7, W: 4x7
    # Indices 0,1,2,3 correspond to A00, A01, A10, A11
    U_strassen = [
        [1, 0, 0, 0, 1, 0, 0], # A00
        [0, 0, 1, 0, 1, 0, -1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, -1, 0]  # A11
    ]
    V_strassen = [
        [1, 1, 0, 0, 0, 1, 0], # B00
        [0, 0, -1, 1, 0, 1, 0], # B01
        [0, 0, 1, -1, 1, 0, 1], # B10
        [1, 0, 0, 0, 1, 0, 1]  # B11
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0], # C01
        [0, 1, 0, 1, 0, 0, 0], # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]

    # Helper to permute rows
    def permute(mat, mapping):
        # mapping: list of indices, new_row[i] = old_row[mapping[i]]
        # Actually, we want to place old values at new positions.
        # Let's build new mat.
        new_mat = [[0] * len(mat[0]) for _ in range(len(mat))]
        for new_idx, old_idx in enumerate(mapping):
            for c in range(len(mat[0])):
                new_mat[new_idx][c] = mat[old_idx][c]
        return new_mat

    # Build M_2,2,4 decomposition
    # U_224: 4x14
    # V_224: 8x14 (B is 2x4)
    # W_224: 8x14 (C is 2x4)
    
    # U is same for both halves (A is shared)
    U_224 = []
    for i in range(4):
        U_224.append(U_strassen[i] + U_strassen[i])
        
    # V_224
    # B_a (cols 0,1) mapping: B00->0, B01->1, B10->2, B11->3
    # B_b (cols 2,3) mapping: B00->4, B01->5, B10->6, B11->7
    V_224 = []
    for i in range(8):
        V_224.append([0] * 14)
        
    # Fill V_224 for first 7 columns (B_a)
    for r in range(4):
        for c in range(7):
            V_224[r][c] = V_strassen[r][c]
            
    # Fill V_224 for next 7 columns (B_b)
    for r in range(4):
        for c in range(7):
            V_224[r+4][c+7] = V_strassen[r][c]

    # W_224
    # C_a (cols 0,1) mapping: C00->0, C01->1, C10->2, C11->3
    # C_b (cols 2,3) mapping: C00->4, C01->5, C10->6, C11->7
    W_224 = []
    for i in range(8):
        W_224.append([0] * 14)
        
    for r in range(4):
        for c in range(7):
            W_224[r][c] = W_strassen[r][c]
            
    for r in range(4):
        for c in range(7):
            W_224[r+4][c+7] = W_strassen[r][c]

    # Build M_2,4,4 decomposition
    # U: 8x28
    # V: 16x28
    # W: 8x28
    
    U = []
    for i in range(8):
        U.append([0] * 28)
        
    V = []
    for i in range(16):
        V.append([0] * 28)
        
    W = []
    for i in range(8):
        W.append([0] * 28)

    # Part 1: A1 * B1
    # A1 indices: 0,1,2,3
    # B1 indices: 0..7
    # C indices: 0..7
    
    for r in range(4):
        for c in range(14):
            U[r][c] = U_224[r][c]
            
    for r in range(8):
        for c in range(14):
            V[r][c] = V_224[r][c]
            
    for r in range(8):
        for c in range(14):
            W[r][c] = W_224[r][c]

    # Part 2: A2 * B2
    # A2 indices: 4,5,6,7
    # B2 indices: 8..15
    # C indices: 0..7 (added to previous)
    
    # Map U_224 rows 0..3 to U rows 4..7
    for r in range(4):
        for c in range(14):
            U[r+4][c+14] = U_224[r][c]
            
    # Map V_224 rows 0..7 to V rows 8..15
    # Note: V_224 indices 0..7 correspond to B1 rows 0..1, cols 0..3
    # V global indices for B2 (rows 2..3) are 8..15
    # B2 is 2x4.
    # V_224 row 0 (B00 of B1) -> Global B20?
    # Wait. B1 is rows 0,1 of B. B2 is rows 2,3 of B.
    # V_224 was built for a 2x4 matrix.
    # Row 0 of V_224 corresponds to Row 0, Col 0 of the 2x4 block.
    # In B1, this is B[0,0]. Global index 0.
    # In B2, this is B[2,0]. Global index 8.
    # So V_224[r] maps to V[r+8].
    
    for r in range(8):
        for c in range(14):
            V[r+8][c+14] = V_224[r][c]
            
    # W is added
    for r in range(8):
        for c in range(14):
            W[r][c+14] = W_224[r][c]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
