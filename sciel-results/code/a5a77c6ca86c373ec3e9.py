import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors definition
    # Each factor: (U_coeffs, V_coeffs, W_coeffs)
    # Indices are local 0..1 for 2x2 matrices
    # U_coeffs: list of (local_idx, coeff)
    
    strassen_U = [
        [(0, 1), (3, 1)],      # M1: X00 + X11
        [(2, 1), (3, 1)],      # M2: X10 + X11
        [(0, 1)],              # M3: X00
        [(3, 1)],              # M4: X11
        [(0, 1), (1, 1)],      # M5: X00 + X01
        [(2, 1), (0, -1)],     # M6: X10 - X00
        [(3, 1), (1, -1)]      # M7: X11 - X01
    ]
    
    strassen_V = [
        [(0, 1), (3, 1)],      # M1: Y00 + Y11
        [(0, 1)],              # M2: Y00
        [(2, 1), (3, -1)],     # M3: Y10 - Y11
        [(0, 1), (2, -1)],     # M4: Y00 - Y10
        [(3, 1)],              # M5: Y11
        [(0, 1), (2, 1)],      # M6: Y00 + Y10
        [(1, 1), (3, 1)]       # M7: Y01 + Y11
    ]
    
    # W_coeffs: list of (local_C_idx, coeff)
    # Z00=M1+M4-M5+M7, Z01=M3+M5, Z10=M2+M4, Z11=M1-M2+M3+M6
    # M1: Z00(1), Z11(1)
    # M2: Z10(1), Z11(-1)
    # M3: Z01(1), Z11(1)
    # M4: Z00(1), Z10(1)
    # M5: Z00(-1), Z01(1)
    # M6: Z11(1)
    # M7: Z00(1)
    
    strassen_W = [
        [(0, 1), (3, 1)],      # M1
        [(2, 1), (3, -1)],     # M2
        [(1, 1), (3, 1)],      # M3
        [(0, 1), (2, 1)],      # M4
        [(0, -1), (1, 1)],     # M5
        [(3, 1)],              # M6
        [(0, 1)]               # M7
    ]
    
    # Blocks definition
    # block: (A_row_offset, A_col_offset, B_row_offset, B_col_offset)
    # A is 2x4. B is 4x6.
    # A offsets:
    # A_L: col 0. A_R: col 2. (Row offset always 0 since n=2)
    # B offsets:
    # B_T: row 0. B_B: row 2.
    # B col offsets: 0, 2, 4.
    
    blocks = [
        # (A_col_off, B_row_off, B_col_off)
        # Note: A_row_off is 0, B_row_off is 0 or 2.
        # A_col_off is 0 or 2.
        # B_col_off is 0, 2, 4.
        
        # Block 0: A_L, B_T, Cols 0-1
        (0, 0, 0),
        # Block 1: A_L, B_T, Cols 2-3
        (0, 0, 2),
        # Block 2: A_L, B_T, Cols 4-5
        (0, 0, 4),
        # Block 3: A_R, B_B, Cols 0-1
        (2, 2, 0),
        # Block 4: A_R, B_B, Cols 2-3
        (2, 2, 2),
        # Block 5: A_R, B_B, Cols 4-5
        (2, 2, 4)
    ]
    
    for b_idx, (a_col, b_row, b_col) in enumerate(blocks):
        for s_idx in range(7):
            col = b_idx * 7 + s_idx
            
            # Update U
            # U shape: (n*m, rank) = (8, 42)
            # Local A index (r, c) -> Global (r, a_col+c)
            # Global index = r*m + (a_col+c) = r*4 + a_col + c
            for (local_idx, coeff) in strassen_U[s_idx]:
                r = local_idx // 2
                c = local_idx % 2
                global_idx = r * m + a_col + c
                U[global_idx][col] += coeff
                
            # Update V
            # V shape: (m*p, rank) = (24, 42)
            # Local B index (r, c) -> Global (b_row+r, b_col+c)
            # Global index = (b_row+r)*p + (b_col+c)
            for (local_idx, coeff) in strassen_V[s_idx]:
                r = local_idx // 2
                c = local_idx % 2
                global_idx = (b_row + r) * p + (b_col + c)
                V[global_idx][col] += coeff
                
            # Update W
            # W shape: (p*n, rank) = (12, 42)
            # Local C index (r, c) -> Global (r, b_col+c)
            # Global W index = k*n + i = (b_col+c)*n + r
            for (local_idx, coeff) in strassen_W[s_idx]:
                r = local_idx // 2
                c = local_idx % 2
                global_i = r
                global_k = b_col + c
                global_idx = global_k * n + global_i
                W[global_idx][col] += coeff
                
    return {"U": U, "V": V, "W": W}
