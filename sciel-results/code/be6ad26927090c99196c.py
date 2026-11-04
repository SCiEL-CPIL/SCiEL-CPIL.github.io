import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # We split the 3x4 matrix A into A1 (2x4) and A2 (1x4).
    # C = A B = [ A1 B ; A2 B ]
    # C1 = A1 B is 2x8.
    # C2 = A2 B is 1x8.
    
    # Rank of C2 (1x4x8) is 32 (schoolbook).
    # Rank of C1 (2x4x8) is 56 (Strassen based).
    # Total rank 88.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)] # 12 x 88
    V = [[0] * rank for _ in range(m * p)] # 32 x 88
    W = [[0] * rank for _ in range(p * n)] # 24 x 88
    
    col = 0
    
    # Part 1: A2 B (1x4x8)
    # A2 is row 2 of A (index 2 in 0-based).
    # B is 4x8.
    # C2 is 1x8.
    # Schoolbook: for j in 0..3, k in 0..7
    for j in range(4):
        for k in range(8):
            # U index: A2 is row 2 of A. In U (n*m x rank), row index is i*m + j.
            # i=2, j=j.
            U[2 * 4 + j][col] = 1
            # V index: B row j, col k. Index j*p + k.
            V[j * 8 + k][col] = 1
            # W index: C2 is row 0 of C (top part? No, C2 is bottom part).
            # C is 3x8. C2 is the last row (i=2).
            # W index is k*n + i. k is col of C, i is row of C.
            # i=2, k=k.
            W[k * 3 + 2][col] = 1
            col += 1
            
    # Part 2: A1 B (2x4x8)
    # A1 is rows 0,1 of A.
    # We split A1 into AL (cols 0,1) and AR (cols 2,3).
    # AL is 2x2. AR is 2x2.
    # B is 4x8. Split into BT (rows 0,1) and BB (rows 2,3).
    # BT is 2x8. BB is 2x8.
    # C1 = AL BT + AR BB.
    # C1 is 2x8.
    
    # We need to compute AL BT and AR BB.
    # Each is a 2x2x8 multiplication.
    # We split the 8 columns of BT and BB into 4 blocks of 2 columns.
    # So we have 4 blocks for BT and 4 for BB.
    # Total 8 blocks of 2x2x2.
    # Each 2x2x2 uses Strassen (rank 7).
    
    # Strassen for 2x2x2:
    # Inputs: X (2x2), Y (2x2).
    # X = [[x11, x12], [x21, x22]]
    # Y = [[y11, y12], [y21, y22]]
    # Z = XY = [[z11, z12], [z21, z22]]
    # 
    # M1 = (x11 + x22)(y11 + y22)
    # M2 = (x21 + x22)y11
    # M3 = x11(y12 - y22)
    # M4 = x22(y21 - y11)
    # M5 = (x11 + x12)y22
    # M6 = (x21 - x11)(y11 + y12)
    # M7 = (x12 - x22)(y21 + y22)
    # 
    # z11 = M1 + M4 - M5 + M7
    # z12 = M3 + M5
    # z21 = M2 + M4
    # z22 = M1 - M2 + M3 + M6
    
    # We need to add terms to U, V, W.
    # For a 2x2x2 block:
    # U corresponds to X.
    # V corresponds to Y.
    # W corresponds to Z.
    
    # Let's define a helper to add a Strassen term.
    # U_idx: index in U for the 2x2 block.
    # V_idx: index in V for the 2x2 block.
    # W_idx: index in W for the 2x2 block.
    
    # The 2x2 block of A1 (rows 0,1) and B (cols 0,1 or 2,3).
    # Wait, A1 is 2x4. AL is cols 0,1. AR is cols 2,3.
    # So AL uses U indices for (i=0, j=0), (i=0, j=1), (i=1, j=0), (i=1, j=1).
    # AR uses U indices for (i=0, j=2), (i=0, j=3), (i=1, j=2), (i=1, j=3).
    
    # BT uses V indices for rows 0,1 of B.
    # BB uses V indices for rows 2,3 of B.
    
    # C1 is 2x8.
    # C1[0, k] and C1[1, k].
    # W indices for C1 are (k=0..7, i=0) and (k=0..7, i=1).
    # W index = k*n + i.
    
    def add_strassen_block(u_offsets, v_offsets, w_offsets, start_col):
        # u_offsets: list of 4 indices into U for the 2x2 block (x11, x12, x21, x22)
        # v_offsets: list of 4 indices into V for the 2x2 block (y11, y12, y21, y22)
        # w_offsets: list of 4 indices into W for the 2x2 block (z11, z12, z21, z22)
        # start_col: current column index
        
        # M1 = (x11 + x22)(y11 + y22)
        U[u_offsets[0]][start_col] = 1
        U[u_offsets[3]][start_col] = 1
        V[v_offsets[0]][start_col] = 1
        V[v_offsets[3]][start_col] = 1
        W[w_offsets[0]][start_col] = 1
        W[w_offsets[3]][start_col] = 1
        W[w_offsets[1]][start_col] = 0 # z12 not in M1
        W[w_offsets[2]][start_col] = 0 # z21 not in M1
        # Wait, W is output.
        # z11 = M1 + M4 - M5 + M7
        # z12 = M3 + M5
        # z21 = M2 + M4
        # z22 = M1 - M2 + M3 + M6
        
        # We need to accumulate coefficients.
        # It is easier to just add the terms directly.
        
        # M1
        # U: x11, x22. V: y11, y22. W: z11, z22.
        # Coeffs for U: 1, 1. V: 1, 1. W: 1, 1.
        # But we need to map to specific entries.
        
        # Let's just iterate through the 7 Strassen terms.
        
        # Term 1: M1 = (x11 + x22)(y11 + y22)
        # U: x11 (1), x22 (1)
        # V: y11 (1), y22 (1)
        # W: z11 (1), z22 (1)
        c = start_col
        U[u_offsets[0]][c] += 1
        U[u_offsets[3]][c] += 1
        V[v_offsets[0]][c] += 1
        V[v_offsets[3]][c] += 1
        W[w_offsets[0]][c] += 1
        W[w_offsets[3]][c] += 1
        c += 1
        
        # Term 2: M2 = (x21 + x22)y11
        # U: x21 (1), x22 (1)
        # V: y11 (1)
        # W: z21 (1), z22 (-1)
        U[u_offsets[2]][c] += 1
        U[u_offsets[3]][c] += 1
        V[v_offsets[0]][c] += 1
        W[w_offsets[2]][c] += 1
        W[w_offsets[3]][c] += -1
        c += 1
        
        # Term 3: M3 = x11(y12 - y22)
        # U: x11 (1)
        # V: y12 (1), y22 (-1)
        # W: z12 (1), z22 (1)
        U[u_offsets[0]][c] += 1
        V[v_offsets[1]][c] += 1
        V[v_offsets[3]][c] += -1
        W[w_offsets[1]][c] += 1
        W[w_offsets[3]][c] += 1
        c += 1
        
        # Term 4: M4 = x22(y21 - y11)
        # U: x22 (1)
        # V: y21 (1), y11 (-1)
        # W: z11 (1), z21 (1)
        U[u_offsets[3]][c] += 1
        V[v_offsets[2]][c] += 1
        V[v_offsets[0]][c] += -1
        W[w_offsets[0]][c] += 1
        W[w_offsets[2]][c] += 1
        c += 1
        
        # Term 5: M5 = (x11 + x12)y22
        # U: x11 (1), x12 (1)
        # V: y22 (1)
        # W: z11 (-1), z12 (1)
        U[u_offsets[0]][c] += 1
        U[u_offsets[1]][c] += 1
        V[v_offsets[3]][c] += 1
        W[w_offsets[0]][c] += -1
        W[w_offsets[1]][c] += 1
        c += 1
        
        # Term 6: M6 = (x21 - x11)(y11 + y12)
        # U: x21 (1), x11 (-1)
        # V: y11 (1), y12 (1)
        # W: z22 (1)
        U[u_offsets[2]][c] += 1
        U[u_offsets[0]][c] += -1
        V[v_offsets[0]][c] += 1
        V[v_offsets[1]][c] += 1
        W[w_offsets[3]][c] += 1
        c += 1
        
        # Term 7: M7 = (x12 - x22)(y21 + y22)
        # U: x12 (1), x22 (-1)
        # V: y21 (1), y22 (1)
        # W: z11 (1)
        U[u_offsets[1]][c] += 1
        U[u_offsets[3]][c] += -1
        V[v_offsets[2]][c] += 1
        V[v_offsets[3]][c] += 1
        W[w_offsets[0]][c] += 1
        c += 1
        
        return c

    # U offsets for AL (cols 0,1 of A1, which are cols 0,1 of A)
    # A1 is rows 0,1.
    # AL: (0,0), (0,1), (1,0), (1,1) -> U indices: 0, 1, 4, 5
    u_AL = [0, 1, 4, 5]
    # AR: (0,2), (0,3), (1,2), (1,3) -> U indices: 2, 3, 6, 7
    u_AR = [2, 3, 6, 7]
    
    # V offsets for BT (rows 0,1 of B)
    # B is 4x8. BT is rows 0,1.
    # For a 2x2 block of B (cols c, c+1), V indices are:
    # (0,c), (0,c+1), (1,c), (1,c+1) -> V indices: 8*c, 8*c+1, 8*c+8, 8*c+9
    # Wait, V index is j*p + k. j is row, k is col.
    # So for row 0, col c: 0*8+c = c.
    # Row 0, col c+1: c+1.
    # Row 1, col c: 8+c.
    # Row 1, col c+1: 8+c+1.
    
    # V offsets for BB (rows 2,3 of B)
    # Row 2, col c: 2*8+c = 16+c.
    # Row 2, col c+1: 16+c+1.
    # Row 3, col c: 3*8+c = 24+c.
    # Row 3, col c+1: 24+c+1.
    
    # W offsets for C1 (2x8)
    # C1 is 2x8.
    # For a 2x2 block of C1 (cols c, c+1), W indices are:
    # (c,0), (c+1,0), (c,1), (c+1,1) -> W indices: c*3+0, (c+1)*3+0, c*3+1, (c+1)*3+1
    # W index is k*n + i. k is col, i is row.
    # So for col c, row 0: c*3.
    # Col c+1, row 0: (c+1)*3.
    # Col c, row 1: c*3+1.
    # Col c+1, row 1: (c+1)*3+1.
    
    # We have 4 blocks for BT and 4 for BB.
    # Block index b from 0 to 3.
    # Cols are 2*b, 2*b+1.
    
    for b in range(4):
        c_start = 2 * b
        
        # BT block
        v_BT = [c_start, c_start + 1, 8 + c_start, 8 + c_start + 1]
        w_C1 = [c_start * 3, (c_start + 1) * 3, c_start * 3 + 1, (c_start + 1) * 3 + 1]
        
        col = add_strassen_block(u_AL, v_BT, w_C1, col)
        
        # BB block
        v_BB = [16 + c_start, 16 + c_start + 1, 24 + c_start, 24 + c_start + 1]
        # W offsets are the same for C1
        col = add_strassen_block(u_AR, v_BB, w_C1, col)
        
    return {"U": U, "V": V, "W": W}
