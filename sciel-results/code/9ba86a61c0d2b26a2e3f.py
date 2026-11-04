"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Decomposition:
    # M_{2,4,7} = M_{2,4,4} + M_{2,4,3}
    # M_{2,4,4} = 4 * M_{2,2,2} (Rank 7 Winograd) -> Rank 28
    # M_{2,4,3} = 2 * M_{2,2,3} (Rank 11: 1x M_{2,2,2} + 1x M_{2,2,1}) -> Rank 22
    # Total Rank = 50
    
    # Winograd 2x2x2 (Rank 7)
    # A: 0:11, 1:12, 2:21, 3:22
    # B: 0:11, 1:12, 2:21, 3:22
    # C: 0:11, 1:12, 2:21, 3:22
    U_wino = [
        [1, 0, 0, 1, 0, 1, 0], # A11
        [0, 1, 0, 1, 0, 0, 1], # A12
        [0, 0, 1, 0, 1, 1, 0], # A21
        [1, 0, 0, 0, 1, 0, 1]  # A22
    ]
    V_wino = [
        [1, 0, 0, 1, 0, 0, 0], # B11
        [0, 0, 1, 0, 0, 0, 1], # B12
        [0, 1, 0, 0, 0, 1, 0], # B21
        [1, 0, 0, 0, 1, 0, 0]  # B22
    ]
    W_wino = [
        [1, 0, 0, 0, 0, 0, 0], # C11 = p1
        [0, 0, 0, 0, 0, 0, 1], # C12 = p7? No.
        [0, 0, 0, 0, 0, 1, 0], # C21 = p6?
        [1, 0, 0, 0, 0, 0, 0]  # C22 = p1
    ]
    # Correct W for Winograd 2x2:
    # C11 = p1 - p4 + p6
    # C12 = p3 + p4 - p7
    # C21 = p2 + p5 - p6
    # C22 = p1 - p2 + p3 + p5 - p7
    W_wino = [
        [1, 0, 0, -1, 0, 1, 0], # C11
        [0, 0, 1, 1, 0, 0, -1], # C12
        [0, 1, 0, 0, 1, -1, 0], # C21
        [1, -1, 1, 0, 1, 0, -1] # C22
    ]
    
    # Schoolbook 2x2x1 (Rank 4)
    # A: 0:11, 1:12, 2:21, 3:22
    # B: 0:11, 1:21 (B is 2x1, so B11, B21)
    # C: 0:11, 1:21 (C is 2x1)
    # C11 = A11 B11 + A12 B21
    # C21 = A21 B11 + A22 B21
    U_221 = [
        [1, 0, 0, 0], # A11
        [0, 1, 0, 0], # A12
        [0, 0, 1, 0], # A21
        [0, 0, 0, 1]  # A22
    ]
    V_221 = [
        [1, 0, 0, 0], # B11
        [0, 0, 1, 0]  # B21
    ]
    W_221 = [
        [1, 1, 0, 0], # C11
        [0, 0, 1, 1]  # C21
    ]
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: M_{2,4,4}
    # A is 2x4. B1 is 4x4.
    # Split A into A1 (2x2, rows 0-1, cols 0-1) and A2 (2x2, rows 0-1, cols 2-3)
    # Split B1 into B11 (2x4, rows 0-1), B12 (2x4, rows 2-3)
    # C1 = A1 B11 + A2 B12
    # A1 B11 is 2x2 x 2x4 -> 2x2 x 2x2 (cols 0-1 of B11) and 2x2 x 2x2 (cols 2-3 of B11)
    
    # Sub-block 1.1: A1 (2x2) x B11_left (2x2) -> C11_left (2x2)
    # A indices: 0,1,2,3
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    
    # Map U_wino to U
    for i in range(4):
        for k in range(7):
            U[i][col+k] = U_wino[i][k]
            V[i][col+k] = V_wino[i][k]
            W[i][col+k] = W_wino[i][k]
    col += 7
    
    # Sub-block 1.2: A1 (2x2) x B11_right (2x2) -> C11_right (2x2)
    # B11_right corresponds to B1 cols 2,3.
    # B indices in V: 2,3,4,5 (since B11 is rows 0,1 of B1, which are cols 0,1,2,3 of B1? No.
    # B1 is 4x4. Rows 0,1 are the top half.
    # B11 is the top-left 2x2. B11_right is top-right 2x2.
    # So for Sub-block 1.2, B indices are 2,3,6,7? No.
    # B1 is 4x4.
    # B11 (top-left): rows 0,1; cols 0,1.
    # B12 (top-right): rows 0,1; cols 2,3.
    # So for Sub-block 1.2, we use B cols 2,3.
    
    for i in range(4):
        for k in range(7):
            U[i][col+k] = U_wino[i][k]
            # V indices for B12: B11 is 0,1; B12 is 2,3?
            # V row index j*p + k. j=0,1. k=0..3.
            # For B11 (cols 0,1), k=0,1.
            # For B12 (cols 2,3), k=2,3.
            # So we map V_wino[i][k] to V[i][col+k] but shifted?
            # No, V_wino is 4x7.
            # We just add to V[i][col+k].
            V[i][col+k] = V_wino[i][k]
            W[i][col+k] = W_wino[i][k]
    col += 7
    
    # Sub-block 1.3: A2 (2x2) x B21 (2x2) -> C21_left (2x2)
    # A2 is rows 0,1; cols 2,3 of A.
    # So U indices are 2,3,4,5.
    # B21 is rows 2,3; cols 0,1 of B1.
    # So V indices are 2*4+0, 2*4+1, 3*4+0, 3*4+1 -> 8,9,12,13.
    
    for i in range(4):
        for k in range(7):
            # A2 indices: 2,3,4,5
            U[2+i][col+k] = U_wino[i][k]
            # B21 indices: 8,9,12,13
            # V_wino[i] corresponds to B11,B12,B21,B22?
            # V_wino rows are B11, B12, B21, B22.
            # So row 0->B11, 1->B12, 2->B21, 3->B22.
            # For B21, we use V_wino[2].
            # But we need to map to V[8+k] and V[12+k]?
            # V is 4x7 for the sub-block.
            # V_wino[0] is for B11 (2x2).
            # V_wino[1] is for B12 (2x2).
            # V_wino[2] is for B21 (2x2).
            # V_wino[3] is for B22 (2x2).
            
            # In this sub-block, we are multiplying A2 by B21.
            # So we use V_wino[2] for the first 2 rows of B21?
            # B21 is 2x2.
            # V_wino[2] is a row of length 7.
            # We add it to V[8][col+k] and V[12][col+k]?
            # No, V_wino[2] is for B21's first row?
            # V_wino is 4x7.
            # Row 0: B11 row 1.
            # Row 1: B11 row 2.
            # Row 2: B12 row 1.
            # Row 3: B12 row 2.
            # Wait, V_wino in my definition:
            # V_wino[0] = [1,0,0,1,0,0,0] -> B11
            # V_wino[1] = [0,0,1,0,0,0,1] -> B12
            # V_wino[2] = [0,1,0,0,0,1,0] -> B21
            # V_wino[3] = [1,0,0,0,1,0,0] -> B22
            
            # So for B21, we use V_wino[2] and V_wino[3]?
            # No, B21 is 2x2.
            # V_wino[2] is for B21 row 1?
            # Let's assume V_wino rows 0,1 are for B11, 2,3 for B12?
            # No, standard Winograd V is 4x7.
            # Row 0: B11
            # Row 1: B12
            # Row 2: B21
            # Row 3: B22
            
            # So for B21, we use Row 2.
            # But B21 has 2 rows.
            # My V_wino definition:
            # V_wino[0] = B11
            # V_wino[1] = B12
            # V_wino[2] = B21
            # V_wino[3] = B22
            
            # This implies B is 4x4? No, B is 2x2 in 2x2x2.
            # So V_wino is 4x7.
            # Row 0: B11
            # Row 1: B12
            # Row 2: B21
            # Row 3: B22
            
            # So for Sub-block 1.3 (A2 x B21), we use:
            # U rows 2,3,4,5 (A2)
            # V rows 8,12 (B21 rows 1,2)
            # W rows 8,12 (C21 rows 1,2)
            
            V[8][col+k] = V_wino[2][k]
            V[12][col+k] = V_wino[3][k] # Wait, V_wino[3] is B22.
            # My V_wino is wrong for this mapping.
            
            # I will abort this manual mapping and use a simpler approach:
            # Just use Schoolbook for everything to be safe.
            pass

    # Fallback to Schoolbook Rank 56
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
