"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for X (2x2)
    # X indices: 00, 01, 10, 11
    # L_X[t] coefficients for X_00, X_01, X_10, X_11
    LX = [
        [1, 0, 0, 1],  # m1
        [0, 0, 1, 1],  # m2
        [1, 0, 0, 0],  # m3
        [0, 0, 0, 1],  # m4
        [1, 1, 0, 0],  # m5
        [-1, 0, 1, 0], # m6
        [0, 1, 0, -1]  # m7
    ]
    
    # Strassen coefficients for Y (2x2)
    # Y indices: 00, 01, 10, 11
    # L_Y[t] coefficients for Y_00, Y_01, Y_10, Y_11
    LY = [
        [1, 0, 0, 1],  # m1
        [1, 0, 0, 0],  # m2
        [0, 1, 0, -1], # m3
        [0, 0, 1, -1], # m4
        [0, 0, 0, 1],  # m5
        [1, 1, 0, 0],  # m6
        [0, 0, 1, 1]   # m7
    ]
    
    # Output coefficients
    # Z_00 = m1 + m4 - m5 + m7
    # Z_01 = m3 + m5
    # Z_10 = m2 + m4
    # Z_11 = m1 - m2 + m3 + m6
    # Order: Z_00, Z_01, Z_10, Z_11
    OUTF = [
        [1, 0, 0, 1, -1, 0, 1],  # Z_00
        [0, 0, 1, 0, 1, 0, 0],   # Z_01
        [0, 1, 0, 1, 0, 0, 0],   # Z_10
        [1, -1, 1, 0, 0, 1, 0]   # Z_11
    ]
    
    def add_strassen_block(col_start, x_map, y_map, z_map):
        for t in range(7):
            col = col_start + t
            # Fill U
            for i in range(4):
                if LX[t][i] != 0:
                    idx = x_map(i)
                    U[idx][col] += LX[t][i]
            # Fill V
            for i in range(4):
                if LY[t][i] != 0:
                    idx = y_map(i)
                    V[idx][col] += LY[t][i]
            # Fill W
            for i in range(4):
                if OUTF[i][t] != 0:
                    idx = z_map(i)
                    W[idx][col] += OUTF[i][t]
    
    # 4 blocks q = 0..3
    for q in range(4):
        # Block 1: A_1 * B^{(q)}_1
        # A_1: cols 0,1 of A. Rows 0,1.
        # x_map: X_ab -> A_a, b. U index a*4 + b
        def x_map_1(a):
            # a is 0..3 representing 00, 01, 10, 11
            # 00: (0,0), 01: (0,1), 10: (1,0), 11: (1,1)
            # Wait, my LX list uses indices 0,1,2,3 for 00, 01, 10, 11?
            # Let's check LX definition.
            # LX[0] = [1, 0, 0, 1] -> X_00 + X_11.
            # So index 0 is X_00, 1 is X_01, 2 is X_10, 3 is X_11.
            # X_ab where a=row, b=col.
            # 00: row 0 col 0. Index 0.
            # 01: row 0 col 1. Index 1.
            # 10: row 1 col 0. Index 2.
            # 11: row 1 col 1. Index 3.
            # So map i (0..3) to (r, c):
            # 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            r = i // 2
            c = i % 2
            return r * 4 + c
        
        # B^{(q)}_1: rows 0,1 of B. Cols 2q, 2q+1.
        # y_map: Y_cd -> B_c, 2q+d. V index c*8 + (2q+d)
        def y_map_1(c_idx):
            c = c_idx // 2
            d = c_idx % 2
            return c * 8 + (2 * q + d)
        
        # Z_ab -> C_a, 2q+b. W index (2q+b)*2 + a
        def z_map_1(z_idx):
            a = z_idx // 2
            b = z_idx % 2
            return (2 * q + b) * 2 + a
        
        add_strassen_block(q * 14, x_map_1, y_map_1, z_map_1)
        
        # Block 2: A_2 * B^{(q)}_2
        # A_2: cols 2,3 of A. Rows 0,1.
        # X_ab -> A_a, 2+b. U index a*4 + (2+b)
        def x_map_2(a):
            r = a // 2
            c = a % 2
            return r * 4 + (2 + c)
        
        # B^{(q)}_2: rows 2,3 of B. Cols 2q, 2q+1.
        # Y_cd -> B_{2+c}, 2q+d. V index (2+c)*8 + (2q+d)
        def y_map_2(c_idx):
            c = c_idx // 2
            d = c_idx % 2
            return (2 + c) * 8 + (2 * q + d)
        
        # Z_ab -> C_a, 2q+b. W index (2q+b)*2 + a
        def z_map_2(z_idx):
            a = z_idx // 2
            b = z_idx % 2
            return (2 * q + b) * 2 + a
        
        add_strassen_block(q * 14 + 7, x_map_2, y_map_2, z_map_2)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
