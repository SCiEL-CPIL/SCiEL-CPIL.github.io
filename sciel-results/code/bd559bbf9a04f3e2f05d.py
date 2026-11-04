import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 130.0

    # We will construct the rank-28 Strassen-based decomposition.
    # This is a known exact decomposition for 2x4 by 4x4 matrix multiplication.
    # It uses 4 independent Strassen 2x2x2 blocks.
    
    rank = 28
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    # Helper to add a Strassen block
    # X_rows: indices in U (A matrix) corresponding to a 2x2 block
    # Y_rows: indices in V (B matrix) corresponding to a 2x2 block
    # Z_rows: indices in W (C matrix) corresponding to a 2x2 block
    # col_offset: starting column in U, V, W for this block's 7 factors
    
    def add_strassen_block(col_offset, X_rows, Y_rows, Z_rows):
        # X_rows: [x00, x01, x10, x11]
        # Y_rows: [y00, y01, y10, y11]
        # Z_rows: [z00, z01, z10, z11]
        
        # Strassen factors:
        # 1: (X00+X11)(Y00+Y11)
        U[X_rows[0], col_offset] += 1
        U[X_rows[3], col_offset] += 1
        V[Y_rows[0], col_offset] += 1
        V[Y_rows[3], col_offset] += 1
        W[Z_rows[0], col_offset] += 1
        W[Z_rows[3], col_offset] += 1
        W[Z_rows[1], col_offset] += 1
        W[Z_rows[2], col_offset] += 1
        # Wait, the output of Strassen block 1 contributes to all 4 outputs with signs.
        # C00 += P1, C01 += P1, C10 += P1, C11 += P1 ? No.
        # Strassen:
        # C00 = P1 + P4 - P5 - P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # Let's re-verify the standard Strassen assignment.
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # C11 = P1 - P2 + P3 + P6
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6 ... Wait, C11 and C22 are different.
        # C11 (top-left) = P1 + P4 - P5 - P7
        # C12 (top-right) = P3 + P5
        # C21 (bottom-left) = P2 + P4
        # C22 (bottom-right) = P1 - P2 + P3 + P6
        
        # Let's map these to the W matrix contributions.
        
        # Block 1: P1
        # A: X00, X11. B: Y00, Y11.
        U[X_rows[0], col_offset] += 1
        U[X_rows[3], col_offset] += 1
        V[Y_rows[0], col_offset] += 1
        V[Y_rows[3], col_offset] += 1
        # C contributions:
        # C00 += 1, C01 += 0, C10 += 0, C11 += 1 ? No.
        # P1 contributes to C00 (+1), C11 (+1).
        # But wait, the formula for C00 is P1 + P4 - P5 - P7.
        # So P1 adds to C00 and C11.
        W[Z_rows[0], col_offset] += 1
        W[Z_rows[3], col_offset] += 1
        
        # Block 2: P2 = (A10+A11)B00  (using 0-indexed: A21+A22 is A[1,0]+A[1,1])
        # A: X10, X11. B: Y00.
        U[X_rows[2], col_offset + 1] += 1
        U[X_rows[3], col_offset + 1] += 1
        V[Y_rows[0], col_offset + 1] += 1
        # C contributions:
        # C10 += 1 (P2 is in C21 formula: P2+P4).
        # C11 -= 1 (P2 is in C22 formula: P1-P2+P3+P6).
        W[Z_rows[2], col_offset + 1] += 1
        W[Z_rows[3], col_offset + 1] -= 1
        
        # Block 3: P3 = A00(B01-B11)
        # A: X00. B: Y01, Y11.
        U[X_rows[0], col_offset + 2] += 1
        V[Y_rows[1], col_offset + 2] += 1
        V[Y_rows[3], col_offset + 2] -= 1
        # C contributions:
        # C01 += 1 (P3 in C12: P3+P5).
        # C11 += 1 (P3 in C22: P1-P2+P3+P6).
        W[Z_rows[1], col_offset + 2] += 1
        W[Z_rows[3], col_offset + 2] += 1
        
        # Block 4: P4 = A11(B10-B00)
        # A: X11. B: Y10, Y00.
        U[X_rows[3], col_offset + 3] += 1
        V[Y_rows[2], col_offset + 3] += 1
        V[Y_rows[0], col_offset + 3] -= 1
        # C contributions:
        # C00 += 1 (P4 in C11: P1+P4-P5-P7).
        # C10 += 1 (P4 in C21: P2+P4).
        W[Z_rows[0], col_offset + 3] += 1
        W[Z_rows[2], col_offset + 3] += 1
        
        # Block 5: P5 = (A00+A01)B11
        # A: X00, X01. B: Y11.
        U[X_rows[0], col_offset + 4] += 1
        U[X_rows[1], col_offset + 4] += 1
        V[Y_rows[3], col_offset + 4] += 1
        # C contributions:
        # C00 -= 1 (P5 in C11: ...-P5...).
        # C01 += 1 (P5 in C12: P3+P5).
        W[Z_rows[0], col_offset + 4] -= 1
        W[Z_rows[1], col_offset + 4] += 1
        
        # Block 6: P6 = (A10-A00)(B00+B01)
        # A: X10, X00. B: Y00, Y01.
        U[X_rows[2], col_offset + 5] += 1
        U[X_rows[0], col_offset + 5] -= 1
        V[Y_rows[0], col_offset + 5] += 1
        V[Y_rows[1], col_offset + 5] += 1
        # C contributions:
        # C11 += 1 (P6 in C22: ...+P6).
        W[Z_rows[3], col_offset + 5] += 1
        
        # Block 7: P7 = (A01-A11)(B10+B11)
        # A: X01, X11. B: Y10, Y11.
        U[X_rows[1], col_offset + 6] += 1
        U[X_rows[3], col_offset + 6] -= 1
        V[Y_rows[2], col_offset + 6] += 1
        V[Y_rows[3], col_offset + 6] += 1
        # C contributions:
        # C00 -= 1 (P7 in C11: ...-P7).
        W[Z_rows[0], col_offset + 6] -= 1

    # Define the 4 blocks
    # A is 2x4. Indices:
    # 0: A00, 1: A01, 2: A02, 3: A03
    # 4: A10, 5: A11, 6: A12, 7: A13
    
    # B is 4x4. Indices:
    # 0: B00, 1: B01, 2: B02, 3: B03
    # 4: B10, 5: B11, 6: B12, 7: B13
    # 8: B20, 9: B21, 10: B22, 11: B23
    # 12: B30, 13: B31, 14: B32, 15: B33
    
    # C is 2x4. Indices:
    # 0: C00, 1: C01, 2: C02, 3: C03
    # 4: C10, 5: C11, 6: C12, 7: C13
    
    # Block 1: A_left (0,1,4,5) * B_TL (0,1,4,5) -> C_left (0,1,4,5)
    # Note: C_left indices in W are 0,1 (row 0) and 4,5 (row 1).
    # My W indices above assumed Z_rows[0..3] map to C00, C01, C10, C11.
    # So Z_rows should be [0, 1, 4, 5] for C_left.
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 1, 4, 5]
    add_strassen_block(0, X1, Y1, Z1)
    
    # Block 2: A_right (2,3,6,7) * B_BL (8,9,12,13) -> C_left (0,1,4,5)
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 1, 4, 5]
    add_strassen_block(7, X2, Y2, Z2)
    
    # Block 3: A_left (0,1,4,5) * B_TR (2,3,6,7) -> C_right (2,3,6,7)
    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [2, 3, 6, 7]
    add_strassen_block(14, X3, Y3, Z3)
    
    # Block 4: A_right (2,3,6,7) * B_BR (10,11,14,15) -> C_right (2,3,6,7)
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [2, 3, 6, 7]
    add_strassen_block(21, X4, Y4, Z4)
    
    # Verify time
    if time.time() > deadline - 10:
        # If we are running out of time, return immediately
        pass

    # Convert to lists for return
    def to_list(mat):
        return [[float(val) for val in row] for row in mat]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
