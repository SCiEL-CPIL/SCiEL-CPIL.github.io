"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition:
    # X (2x4) = [X1 (2x2), X2 (2x2)]
    # Y (4x7) = [Y1 (2x7); Y2 (2x7)]
    # Z = X1 Y1 + X2 Y2
    # Y1 (2x7) is split into Y11 (2x2), Y12 (2x2), Y13 (2x2), Y14 (2x1)
    # Y2 (2x7) is split into Y21 (2x2), Y22 (2x2), Y23 (2x2), Y24 (2x1)
    
    # Rank contributions:
    # X1 Y11 (2x2x2): Rank 7
    # X1 Y12 (2x2x2): Rank 7
    # X1 Y13 (2x2x2): Rank 7
    # X1 Y14 (2x2x1): Rank 4
    # X2 Y21 (2x2x2): Rank 7
    # X2 Y22 (2x2x2): Rank 7
    # X2 Y23 (2x2x2): Rank 7
    # X2 Y24 (2x2x1): Rank 4
    # Total Rank = 7*6 + 4*2 = 42 + 8 = 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)] # 8 x 50
    V = [[0] * rank for _ in range(m * p)] # 28 x 50
    W = [[0] * rank for _ in range(p * n)] # 14 x 50
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_222(x_offset, y_offset, w_offset):
        # x_offset: index of first row of X block (0 or 2)
        # y_offset: index of first col of Y block (0, 2, 4, 6)
        # w_offset: index of first col of Z block (0, 2, 4, 6)
        nonlocal col
        # Strassen products for A (2x2) at X[x_offset:x_offset+2, 0:2] 
        # and B (2x2) at Y[0:2, y_offset:y_offset+2]
        # Note: X is 2x4. X1 is cols 0-1. X2 is cols 2-3.
        # So for X1, x_offset is 0. For X2, x_offset is 2.
        # Y1 is rows 0-1. Y2 is rows 2-3.
        # Wait, my split was X=[X1, X2] (cols) and Y=[Y1; Y2] (rows).
        # X1 is 2x2 (cols 0-1). X2 is 2x2 (cols 2-3).
        # Y1 is 2x7 (rows 0-1). Y2 is 2x7 (rows 2-3).
        
        # In U, rows correspond to X indices (0..7).
        # X1 uses rows 0,1 (i=0,1; j=0,1 -> idx 0,1,2,3? No, U is n*m x rank).
        # U index: i*m + j.
        # For X1 (i=0,1; j=0,1): indices 0, 1, 2, 3.
        # For X2 (i=0,1; j=2,3): indices 4, 5, 6, 7.
        
        # In V, rows correspond to Y indices (0..27).
        # V index: j*p + k.
        # For Y1 (j=0,1; k=0..6): indices 0..6, 7..13.
        # For Y2 (j=2,3; k=0..6): indices 14..20, 21..27.
        
        # In W, rows correspond to Z indices (0..13).
        # W index: k*n + i.
        # Z is 2x7.
        # Z1 (from X1 Y1) is 2x7. Indices 0..13.
        # Z2 (from X2 Y2) is 2x7. Indices 0..13.
        # We sum them. So W offsets are the same for both.
        
        # Strassen 2x2x2:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # U values for A (2x2)
        # a11 at x_offset + 0
        # a12 at x_offset + 1
        # a21 at x_offset + 2
        # a22 at x_offset + 3
        
        # V values for B (2x2)
        # b11 at y_offset + 0
        # b12 at y_offset + 1
        # b21 at y_offset + 2
        # b22 at y_offset + 3
        
        # W values for C (2x2)
        # c11 at w_offset + 0
        # c12 at w_offset + 1
        # c21 at w_offset + 2
        # c22 at w_offset + 3
        
        # M1
        U[x_offset + 0][col] = 1
        U[x_offset + 3][col] = 1
        V[y_offset + 0][col] = 1
        V[y_offset + 3][col] = 1
        W[w_offset + 0][col] = 1
        W[w_offset + 3][col] = 1
        col += 1
        
        # M2
        U[x_offset + 2][col] = 1
        U[x_offset + 3][col] = 1
        V[y_offset + 0][col] = 1
        W[w_offset + 2][col] = 1
        W[w_offset + 3][col] = 1
        col += 1
        
        # M3
        U[x_offset + 0][col] = 1
        V[y_offset + 1][col] = 1
        V[y_offset + 3][col] = -1
        W[w_offset + 1][col] = 1
        W[w_offset + 3][col] = 1
        col += 1
        
        # M4
        U[x_offset + 3][col] = 1
        V[y_offset + 2][col] = 1
        V[y_offset + 0][col] = -1
        W[w_offset + 0][col] = 1
        W[w_offset + 2][col] = 1
        col += 1
        
        # M5
        U[x_offset + 0][col] = 1
        U[x_offset + 1][col] = 1
        V[y_offset + 3][col] = 1
        W[w_offset + 0][col] = -1
        W[w_offset + 1][col] = 1
        col += 1
        
        # M6
        U[x_offset + 2][col] = 1
        U[x_offset + 0][col] = -1
        V[y_offset + 0][col] = 1
        V[y_offset + 1][col] = 1
        W[w_offset + 3][col] = 1
        col += 1
        
        # M7
        U[x_offset + 1][col] = 1
        U[x_offset + 3][col] = -1
        V[y_offset + 2][col] = 1
        V[y_offset + 3][col] = 1
        W[w_offset + 0][col] = 1
        W[w_offset + 3][col] = -1
        col += 1
        
    # Helper to add a 2x2x1 block
    def add_221(x_offset, y_offset, w_offset):
        # A (2x2), B (2x1).
        # b11 at y_offset + 0
        # b21 at y_offset + 2 (since B is 2x1, indices are j*p+k. 
        # Wait, Y is 4x7.
        # For Y1 (rows 0-1), y_offset is the column start.
        # B is 2x1. b11 is at (0, y_offset). b21 is at (1, y_offset).
        # V index: j*p + k.
        # b11: j=0, k=y_offset -> idx y_offset.
        # b21: j=1, k=y_offset -> idx p + y_offset = 7 + y_offset.
        
        nonlocal col
        # P1 = a11 b11
        U[x_offset + 0][col] = 1
        V[y_offset][col] = 1
        W[w_offset + 0][col] = 1
        col += 1
        
        # P2 = a12 b21
        U[x_offset + 1][col] = 1
        V[7 + y_offset][col] = 1
        W[w_offset + 0][col] = 1
        col += 1
        
        # P3 = a21 b11
        U[x_offset + 2][col] = 1
        V[y_offset][col] = 1
        W[w_offset + 2][col] = 1
        col += 1
        
        # P4 = a22 b21
        U[x_offset + 3][col] = 1
        V[7 + y_offset][col] = 1
        W[w_offset + 2][col] = 1
        col += 1

    # Process X1 (x_offset=0) with Y1 (rows 0-1)
    # Y1 columns: 0-1 (222), 2-3 (222), 4-5 (222), 6 (221)
    add_strassen_222(0, 0, 0)
    add_strassen_222(0, 2, 2)
    add_strassen_222(0, 4, 4)
    add_221(0, 6, 6)
    
    # Process X2 (x_offset=4) with Y2 (rows 2-3)
    # Y2 is rows 2-3.
    # V index for Y2: j=2,3.
    # b11 for Y2 block: j=2, k=y_offset. V idx = 2*7 + y_offset = 14 + y_offset.
    # b21 for Y2 block: j=3, k=y_offset. V idx = 3*7 + y_offset = 21 + y_offset.
    
    # We need to adjust the helpers or pass offsets correctly.
    # The helpers above assumed Y starts at row 0.
    # Let's generalize the helpers to take v_row_offset.
    
    # Re-implementing with explicit indices is safer.
    
    # Reset
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_222_gen(x_off, v_off, w_off):
        # x_off: U row offset (0 or 4)
        # v_off: V row offset for the 2x2 block (0, 2, 4, 6, 14, 16, 18, 22)
        # w_off: W col offset (0, 2, 4, 6)
        nonlocal col
        # A indices: x_off, x_off+1, x_off+2, x_off+3
        # B indices: v_off, v_off+1, v_off+7, v_off+8 (since p=7)
        # C indices: w_off, w_off+1, w_off+2, w_off+3
        
        # M1
        U[x_off][col] = 1; U[x_off+3][col] = 1
        V[v_off][col] = 1; V[v_off+7][col] = 1
        W[w_off][col] = 1; W[w_off+3][col] = 1
        col += 1
        # M2
        U[x_off+2][col] = 1; U[x_off+3][col] = 1
        V[v_off][col] = 1
        W[w_off+2][col] = 1; W[w_off+3][col] = 1
        col += 1
        # M3
        U[x_off][col] = 1
        V[v_off+1][col] = 1; V[v_off+8][col] = -1
        W[w_off+1][col] = 1; W[w_off+3][col] = 1
        col += 1
        # M4
        U[x_off+3][col] = 1
        V[v_off+7][col] = 1; V[v_off][col] = -1
        W[w_off][col] = 1; W[w_off+2][col] = 1
        col += 1
        # M5
        U[x_off][col] = 1; U[x_off+1][col] = 1
        V[v_off+8][col] = 1
        W[w_off][col] = -1; W[w_off+1][col] = 1
        col += 1
        # M6
        U[x_off+2][col] = 1; U[x_off][col] = -1
        V[v_off][col] = 1; V[v_off+1][col] = 1
        W[w_off+3][col] = 1
        col += 1
        # M7
        U[x_off+1][col] = 1; U[x_off+3][col] = -1
        V[v_off+7][col] = 1; V[v_off+8][col] = 1
        W[w_off][col] = 1; W[w_off+3][col] = -1
        col += 1

    def add_221_gen(x_off, v_off, w_off):
        # x_off: U row offset
        # v_off: V row offset for the 2x1 block (6, 22)
        # w_off: W col offset (6)
        nonlocal col
        # A indices: x_off, x_off+1, x_off+2, x_off+3
        # B indices: v_off, v_off+7
        # C indices: w_off, w_off+2
        
        # P1
        U[x_off][col] = 1
        V[v_off][col] = 1
        W[w_off][col] = 1
        col += 1
        # P2
        U[x_off+1][col] = 1
        V[v_off+7][col] = 1
        W[w_off][col] = 1
        col += 1
        # P3
        U[x_off+2][col] = 1
        V[v_off][col] = 1
        W[w_off+2][col] = 1
        col += 1
        # P4
        U[x_off+3][col] = 1
        V[v_off+7][col] = 1
        W[w_off+2][col] = 1
        col += 1

    # X1 (x_off=0) with Y1 (v_off 0, 2, 4, 6)
    add_strassen_222_gen(0, 0, 0)
    add_strassen_222_gen(0, 2, 2)
    add_strassen_222_gen(0, 4, 4)
    add_221_gen(0, 6, 6)
    
    # X2 (x_off=4) with Y2 (v_off 14, 16, 18, 22)
    add_strassen_222_gen(4, 14, 0)
    add_strassen_222_gen(4, 16, 2)
    add_strassen_222_gen(4, 18, 4)
    add_221_gen(4, 22, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
