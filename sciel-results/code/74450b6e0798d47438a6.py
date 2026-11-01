import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # A_L B_T: 3 * R(M222) + 1 * R(M221) = 3*7 + 4 = 25
    # A_R B_B: 3 * R(M222) + 1 * R(M221) = 3*7 + 4 = 25
    # Total Rank = 50
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 50
    V = [[0] * rank for _ in range(m * p)] # 28 x 50
    W = [[0] * rank for _ in range(p * n)] # 14 x 50
    
    col = 0
    
    # Helper to add a Strassen term for 2x2 * 2x2
    # X is 2x2, Y is 2x2
    # Indices for X: i,j in 0..1
    # Indices for Y: j,k in 0..1
    # Output Z: i,k
    def add_strassen_222(X_base, Y_base, Z_base):
        nonlocal col
        # P1 = (X11+X22)(Y11+Y22)
        U[X_base + 0][col] += 1
        U[X_base + 3][col] += 1
        V[Y_base + 0][col] += 1
        V[Y_base + 3][col] += 1
        W[Z_base + 0][col] += 1
        W[Z_base + 2][col] += 1
        col += 1
        
        # P2 = (X21+X22)Y11
        U[X_base + 2][col] += 1
        U[X_base + 3][col] += 1
        V[Y_base + 0][col] += 1
        W[Z_base + 1][col] += 1
        W[Z_base + 3][col] += 1
        col += 1
        
        # P3 = X11(Y12-Y22)
        U[X_base + 0][col] += 1
        V[Y_base + 1][col] += 1
        V[Y_base + 3][col] += -1
        W[Z_base + 0][col] += 1
        W[Z_base + 2][col] += 1
        col += 1
        
        # P4 = X22(Y21-Y11)
        U[X_base + 3][col] += 1
        V[Y_base + 2][col] += 1
        V[Y_base + 0][col] += -1
        W[Z_base + 1][col] += 1
        W[Z_base + 3][col] += 1
        col += 1
        
        # P5 = (X11+X12)Y22
        U[X_base + 0][col] += 1
        U[X_base + 1][col] += 1
        V[Y_base + 3][col] += 1
        W[Z_base + 0][col] += 1
        W[Z_base + 2][col] += 1
        col += 1
        
        # P6 = (X21-X11)(Y11+Y12)
        U[X_base + 2][col] += 1
        U[X_base + 0][col] += -1
        V[Y_base + 0][col] += 1
        V[Y_base + 1][col] += 1
        W[Z_base + 1][col] += 1
        W[Z_base + 3][col] += 1
        col += 1
        
        # P7 = (X12-X22)(Y21+Y22)
        U[X_base + 1][col] += 1
        U[X_base + 3][col] += -1
        V[Y_base + 2][col] += 1
        V[Y_base + 3][col] += 1
        W[Z_base + 0][col] += 1
        W[Z_base + 2][col] += 1
        col += 1

    # Helper to add a term for 2x2 * 2x1
    # X is 2x2, Y is 2x1
    def add_strassen_221(X_base, Y_base, Z_base):
        nonlocal col
        # Q1 = X11 Y1
        U[X_base + 0][col] += 1
        V[Y_base + 0][col] += 1
        W[Z_base + 0][col] += 1
        col += 1
        
        # Q2 = X12 Y2
        U[X_base + 1][col] += 1
        V[Y_base + 1][col] += 1
        W[Z_base + 0][col] += 1
        col += 1
        
        # Q3 = X21 Y1
        U[X_base + 2][col] += 1
        V[Y_base + 0][col] += 1
        W[Z_base + 1][col] += 1
        col += 1
        
        # Q4 = X22 Y2
        U[X_base + 3][col] += 1
        V[Y_base + 1][col] += 1
        W[Z_base + 1][col] += 1
        col += 1

    # A is 2x4. Split into A_L (2x2) and A_R (2x2).
    # A_L indices in U: 0,1,2,3 (row 0: 0,1; row 1: 2,3)
    # A_R indices in U: 4,5,6,7 (row 0: 4,5; row 1: 6,7)
    
    # B is 4x7. Split into B_T (2x7) and B_B (2x7).
    # B_T rows 0,1. B_B rows 2,3.
    # B_T cols 0..6.
    # B_B cols 0..6.
    
    # Process A_L * B_T
    # B_T is 2x7. Split into B_T12 (2x2), B_T34 (2x2), B_T56 (2x2), B_T7 (2x1).
    # B_T12: Y_base 0 (cols 0,1)
    # B_T34: Y_base 2 (cols 2,3)
    # B_T56: Y_base 4 (cols 4,5)
    # B_T7:  Y_base 6 (col 6)
    
    # Z for A_L B_T is 2x7.
    # Z_base for 2x2 block: 
    # C is 2x7.
    # C_L12: rows 0,1, cols 0,1 -> Z_base 0
    # C_L34: rows 0,1, cols 2,3 -> Z_base 2
    # C_L56: rows 0,1, cols 4,5 -> Z_base 4
    # C_L7:  rows 0,1, col 6    -> Z_base 6 (1x1? No, 2x1)
    
    add_strassen_222(0, 0, 0)   # A_L * B_T12 -> C_L12
    add_strassen_222(0, 2, 2)   # A_L * B_T34 -> C_L34
    add_strassen_222(0, 4, 4)   # A_L * B_T56 -> C_L56
    add_strassen_221(0, 6, 6)   # A_L * B_T7  -> C_L7
    
    # Process A_R * B_B
    # A_R base 4
    # B_B is 2x7.
    # B_B12: Y_base 2*7 + 0 = 14
    # B_B34: Y_base 2*7 + 2 = 16
    # B_B56: Y_base 2*7 + 4 = 18
    # B_B7:  Y_base 2*7 + 6 = 20
    
    # Z for A_R B_B is 2x7.
    # C_R12: rows 0,1, cols 0,1 -> Z_base 0 (same as C_L12, they add up)
    # Wait, C = C_L + C_R.
    # So Z_base is the same for both.
    
    add_strassen_222(4, 14, 0)  # A_R * B_B12 -> C_R12 (adds to C_L12)
    add_strassen_222(4, 16, 2)  # A_R * B_B34 -> C_R34
    add_strassen_222(4, 18, 4)  # A_R * B_B56 -> C_R56
    add_strassen_221(4, 20, 6)  # A_R * B_B7  -> C_R7
    
    return {"U": U, "V": V, "W": W}
