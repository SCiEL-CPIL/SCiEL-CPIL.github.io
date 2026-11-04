import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank: 50
    # Decomposition:
    # Split A (2x4) into AL (2x2) and AR (2x2)
    # Split B (4x7) into BT (2x7) and BB (2x7)
    # C = AL*BT + AR*BB
    # BT and BB are 2x7. Split each into three 2x2 and one 2x1.
    # Each 2x2*2x2 is Rank 7 (Strassen).
    # Each 2x2*2x1 is Rank 4 (Schoolbook).
    # Total Rank = 2 * (3*7 + 4) = 2 * 25 = 50.
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add Strassen 2x2 * 2x2
    # a_indices: indices of A elements (0..3 for 2x2)
    # b_indices: indices of B elements (0..3 for 2x2)
    # c_indices: indices of C elements (0..3 for 2x2)
    # offset: current column in U, V, W
    def add_strassen_2x2(a_idx, b_idx, c_idx, offset):
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0], offset] += 1
        U[a_idx[3], offset] += 1
        V[b_idx[0], offset] += 1
        V[b_idx[3], offset] += 1
        W[c_idx[0], offset] += 1
        W[c_idx[3], offset] += 1
        
        # M2 = (a10+a11)b00
        U[a_idx[2], offset+1] += 1
        U[a_idx[3], offset+1] += 1
        V[b_idx[0], offset+1] += 1
        W[c_idx[2], offset+1] += 1
        W[c_idx[3], offset+1] -= 1
        
        # M3 = a00(b10-b11)
        U[a_idx[0], offset+2] += 1
        V[b_idx[2], offset+2] += 1
        V[b_idx[3], offset+2] -= 1
        W[c_idx[1], offset+2] += 1
        W[c_idx[2], offset+2] += 1
        
        # M4 = a11(b01-b00)
        U[a_idx[3], offset+3] += 1
        V[b_idx[1], offset+3] += 1
        V[b_idx[0], offset+3] -= 1
        W[c_idx[0], offset+3] += 1
        W[c_idx[2], offset+3] += 1
        
        # M5 = (a00+a01)b11
        U[a_idx[0], offset+4] += 1
        U[a_idx[1], offset+4] += 1
        V[b_idx[3], offset+4] += 1
        W[c_idx[0], offset+4] -= 1
        W[c_idx[1], offset+4] += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2], offset+5] += 1
        U[a_idx[0], offset+5] -= 1
        V[b_idx[0], offset+5] += 1
        V[b_idx[1], offset+5] += 1
        W[c_idx[3], offset+5] += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1], offset+6] += 1
        U[a_idx[3], offset+6] -= 1
        V[b_idx[2], offset+6] += 1
        V[b_idx[3], offset+6] += 1
        W[c_idx[0], offset+6] += 1
        W[c_idx[1], offset+6] -= 1
        
        return offset + 7

    # Helper to add Schoolbook 2x2 * 2x1
    # a_indices: indices of A elements (0..3 for 2x2)
    # b_indices: indices of B elements (0..1 for 2x1)
    # c_indices: indices of C elements (0..1 for 2x1)
    # offset: current column
    def add_schoolbook_2x1(a_idx, b_idx, c_idx, offset):
        # z00 = a00*b0 + a01*b1
        # z10 = a10*b0 + a11*b1
        
        # Term 1: a00*b0 -> c0
        U[a_idx[0], offset] += 1
        V[b_idx[0], offset] += 1
        W[c_idx[0], offset] += 1
        
        # Term 2: a01*b1 -> c0
        U[a_idx[1], offset+1] += 1
        V[b_idx[1], offset+1] += 1
        W[c_idx[0], offset+1] += 1
        
        # Term 3: a10*b0 -> c1
        U[a_idx[2], offset+2] += 1
        V[b_idx[0], offset+2] += 1
        W[c_idx[1], offset+2] += 1
        
        # Term 4: a11*b1 -> c1
        U[a_idx[3], offset+3] += 1
        V[b_idx[1], offset+3] += 1
        W[c_idx[1], offset+3] += 1
        
        return offset + 4

    # Mapping of global indices to local block indices
    # A is 2x4. Rows 0,1. Cols 0,1,2,3.
    # AL: A[0:2, 0:2]. Local (r, c) -> Global (r, c)
    # AR: A[0:2, 2:4]. Local (r, c) -> Global (r, c+2)
    
    # B is 4x7. Rows 0,1,2,3. Cols 0..6.
    # BT: B[0:2, 0:7]. Local (r, c) -> Global (r, c)
    # BB: B[2:4, 0:7]. Local (r, c) -> Global (r+2, c)
    
    # C is 2x7. Rows 0,1. Cols 0..6.
    # C1 = AL*BT. Local (r, c) -> Global (r, c)
    # C2 = AR*BB. Local (r, c) -> Global (r, c)
    
    # Indices for 2x2 block of A (local r,c in 0..1)
    # Global index in A (2x4): row r, col c (for AL) or c+2 (for AR)
    # U index: row * m + col = r * 4 + col
    
    # Indices for 2x2 block of B (local r,c in 0..1)
    # Global index in B (4x7): row r (for BT) or r+2 (for BB), col c
    # V index: row * p + col = row * 7 + col
    
    # Indices for 2x2 block of C (local r,c in 0..1)
    # Global index in C (2x7): row r, col c
    # W index: col * n + row = col * 2 + row
    
    # Process AL * BT
    # BT cols 0,1 -> 2x2 block
    a_AL = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # a00, a01, a10, a11
    b_BT_01 = [0*7+0, 0*7+1, 1*7+0, 1*7+1] # b00, b01, b10, b11
    c_C1_01 = [0*2+0, 0*2+1, 1*2+0, 1*2+1] # c00, c01, c10, c11
    offset = add_strassen_2x2(a_AL, b_BT_01, c_C1_01, offset)
    
    # BT cols 2,3 -> 2x2 block
    b_BT_23 = [0*7+2, 0*7+3, 1*7+2, 1*7+3]
    c_C1_23 = [2*2+0, 2*2+1, 3*2+0, 3*2+1]
    offset = add_strassen_2x2(a_AL, b_BT_23, c_C1_23, offset)
    
    # BT cols 4,5 -> 2x2 block
    b_BT_45 = [0*7+4, 0*7+5, 1*7+4, 1*7+5]
    c_C1_45 = [4*2+0, 4*2+1, 5*2+0, 5*2+1]
    offset = add_strassen_2x2(a_AL, b_BT_45, c_C1_45, offset)
    
    # BT col 6 -> 2x1 block
    b_BT_6 = [0*7+6, 1*7+6]
    c_C1_6 = [6*2+0, 6*2+1]
    offset = add_schoolbook_2x1(a_AL, b_BT_6, c_C1_6, offset)
    
    # Process AR * BB
    # AR: A[0:2, 2:4]
    a_AR = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
    
    # BB rows 2,3 (global 2,3), cols 0,1
    b_BB_01 = [2*7+0, 2*7+1, 3*7+0, 3*7+1]
    c_C2_01 = [0*2+0, 0*2+1, 1*2+0, 1*2+1] # Same C indices as C1
    offset = add_strassen_2x2(a_AR, b_BB_01, c_C2_01, offset)
    
    # BB rows 2,3, cols 2,3
    b_BB_23 = [2*7+2, 2*7+3, 3*7+2, 3*7+3]
    c_C2_23 = [2*2+0, 2*2+1, 3*2+0, 3*2+1]
    offset = add_strassen_2x2(a_AR, b_BB_23, c_C2_23, offset)
    
    # BB rows 2,3, cols 4,5
    b_BB_45 = [2*7+4, 2*7+5, 3*7+4, 3*7+5]
    c_C2_45 = [4*2+0, 4*2+1, 5*2+0, 5*2+1]
    offset = add_strassen_2x2(a_AR, b_BB_45, c_C2_45, offset)
    
    # BB rows 2,3, col 6
    b_BB_6 = [2*7+6, 3*7+6]
    c_C2_6 = [6*2+0, 6*2+1]
    offset = add_schoolbook_2x1(a_AR, b_BB_6, c_C2_6, offset)
    
    # Final check: offset should be 50
    if offset != rank:
        raise ValueError(f"Offset {offset} does not match rank {rank}")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
