import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Rank 56 solution for 2x4 * 4x8
    # Decomposition based on 8 independent 2x2x2 Strassen blocks
    # 4 blocks for A_left (cols 0-1) * B_top (rows 0-1)
    # 4 blocks for A_right (cols 2-3) * B_bottom (rows 2-3)
    # Each 2x2x2 block has rank 7.
    # Total rank = 8 * 7 = 56.
    
    n, m, p = 2, 4, 8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen matrices for 2x2 multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Coefficients for A (U)
    # Indices: [A11, A12, A21, A22]
    # But A is 2x4. We split into Left (0-1) and Right (2-3).
    # For Left part, indices in U are 0,1 (row 0) and 2,3 (row 1) of the 2x2 block?
    # U shape is (8, 56).
    # A indices: 0..7.
    # A_left: 0,1 (row 0), 2,3 (row 1) ?
    # A is 2x4. Row 0: 0,1,2,3. Row 1: 4,5,6,7.
    # A_left: cols 0,1. Indices 0,1 (row 0), 4,5 (row 1).
    # A_right: cols 2,3. Indices 2,3 (row 0), 6,7 (row 1).
    
    # Let's map the 2x2 Strassen variables to the global indices.
    # For a block (a_block, b_block) where a_block is 0 (left) or 1 (right)
    # and b_block is 0..3 (column block of B).
    
    # Strassen A-combos (local 2x2 indices 00, 01, 10, 11)
    # 00: A11, 01: A12, 10: A21, 11: A22
    A_combos = [
        [1, 0, 0, 1],  # M1: A11+A22
        [0, 0, 1, 1],  # M2: A21+A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11+A12
        [-1, 0, 1, 0], # M6: A21-A11
        [0, 1, 0, -1]  # M7: A12-A22
    ]
    
    # Strassen B-combos (local 2x2 indices 00, 01, 10, 11)
    B_combos = [
        [1, 0, 0, 1],  # M1: B11+B22
        [1, 0, 0, 0],  # M2: B11
        [0, 1, 0, -1], # M3: B12-B22
        [0, 0, 1, -1], # M4: B21-B11
        [0, 0, 0, 1],  # M5: B22
        [1, 1, 0, 0],  # M6: B11+B12
        [0, 0, 1, 1]   # M7: B21+B22
    ]
    
    # Strassen C-combos (local 2x2 indices 00, 01, 10, 11)
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 + M7
    C_combos = [
        [1, 0, 0, -1, 0, 1, -1],  # C11
        [0, 1, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, 0, -1, 0, 1, 0, 1]    # C22
    ]
    
    # Mapping of global indices
    # A: 2 rows, 4 cols.
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # A_left (cols 0,1):
    #   Local 00 (A11) -> Global 0
    #   Local 01 (A12) -> Global 1
    #   Local 10 (A21) -> Global 4
    #   Local 11 (A22) -> Global 5
    # A_right (cols 2,3):
    #   Local 00 (A11) -> Global 2
    #   Local 01 (A12) -> Global 3
    #   Local 10 (A21) -> Global 6
    #   Local 11 (A22) -> Global 7
    
    # B: 4 rows, 8 cols.
    # Rows 0,1 are B_top. Rows 2,3 are B_bottom.
    # Cols 0..7.
    # For block g (0..3), cols are 2g, 2g+1.
    # B_top (rows 0,1):
    #   Local 00 (B11) -> Global Row 0, Col 2g -> Index 0*8 + 2g = 2g
    #   Local 01 (B12) -> Global Row 0, Col 2g+1 -> Index 2g+1
    #   Local 10 (B21) -> Global Row 1, Col 2g -> Index 1*8 + 2g = 8+2g
    #   Local 11 (B22) -> Global Row 1, Col 2g+1 -> Index 8+2g+1
    # B_bottom (rows 2,3):
    #   Local 00 (B11) -> Global Row 2, Col 2g -> Index 2*8 + 2g = 16+2g
    #   Local 01 (B12) -> Global Row 2, Col 2g+1 -> Index 16+2g+1
    #   Local 10 (B21) -> Global Row 3, Col 2g -> Index 3*8 + 2g = 24+2g
    #   Local 11 (B22) -> Global Row 3, Col 2g+1 -> Index 24+2g+1
    
    # C: 2 rows, 8 cols.
    # Row 0: 0..7
    # Row 1: 8..15
    # For block g (0..3), cols are 2g, 2g+1.
    # C11 (Local 00) -> Global Row 0, Col 2g -> Index 2g
    # C12 (Local 01) -> Global Row 0, Col 2g+1 -> Index 2g+1
    # C21 (Local 10) -> Global Row 1, Col 2g -> Index 8+2g
    # C22 (Local 11) -> Global Row 1, Col 2g+1 -> Index 8+2g+1
    
    r = 0
    for a_block in range(2): # 0: Left, 1: Right
        for g in range(4): # 0..3
            for s in range(7): # Strassen term 0..6
                # Fill U
                if a_block == 0:
                    a_indices = [0, 1, 4, 5]
                else:
                    a_indices = [2, 3, 6, 7]
                
                for k in range(4):
                    U[a_indices[k]][r] = A_combos[s][k]
                
                # Fill V
                if a_block == 0:
                    # B_top
                    b_indices = [2*g, 2*g+1, 8+2*g, 8+2*g+1]
                else:
                    # B_bottom
                    b_indices = [16+2*g, 16+2*g+1, 24+2*g, 24+2*g+1]
                
                for k in range(4):
                    V[b_indices[k]][r] = B_combos[s][k]
                
                # Fill W
                c_indices = [2*g, 2*g+1, 8+2*g, 8+2*g+1]
                
                for k in range(4):
                    W[c_indices[k]][r] = C_combos[k][s]
                
                r += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
