import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    """
    Rank-28 exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    Uses 4 independent 2x2 Strassen blocks (7 multiplications each).
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    def add_strassen_block(block_id, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Add a Strassen 2x2*2x2 block to the decomposition.
        X = A[a_rows, a_cols] (2x2)
        Y = B[b_rows, b_cols] (2x2)
        Z = C[c_rows, c_cols] (2x2)
        
        Strassen formulas:
        M1 = (X00+X11)*(Y00+Y11)
        M2 = (X10+X11)*Y00
        M3 = X00*(Y01-Y11)
        M4 = X11*(Y10-Y00)
        M5 = (X00+X01)*Y11
        M6 = (X10-X00)*(Y00+Y01)
        M7 = (X01-X11)*(Y10+Y11)
        
        Z00 = M1 + M4 - M5 + M7
        Z01 = M3 + M5
        Z10 = M2 + M4
        Z11 = M1 - M2 + M3 + M6
        """
        base = block_id * 7
        
        # U indices: row = a_row*m + a_col
        # V indices: row = b_row*p + b_col
        # W indices: col = c_col*n + c_row  (official trace convention W[k*n+i, r])
        
        # M1: (X00+X11)*(Y00+Y11)
        col = base + 0
        U[a_rows[0]*m + a_cols[0]][col] += 1
        U[a_rows[1]*m + a_cols[1]][col] += 1
        V[b_rows[0]*p + b_cols[0]][col] += 1
        V[b_rows[1]*p + b_cols[1]][col] += 1
        # Z00 += M1, Z11 += M1
        W[c_cols[0]*n + c_rows[0]][col] += 1
        W[c_cols[1]*n + c_rows[1]][col] += 1
        
        # M2: (X10+X11)*Y00
        col = base + 1
        U[a_rows[1]*m + a_cols[0]][col] += 1
        U[a_rows[1]*m + a_cols[1]][col] += 1
        V[b_rows[0]*p + b_cols[0]][col] += 1
        # Z10 += M2, Z11 -= M2
        W[c_cols[0]*n + c_rows[1]][col] += 1
        W[c_cols[1]*n + c_rows[1]][col] -= 1
        
        # M3: X00*(Y01-Y11)
        col = base + 2
        U[a_rows[0]*m + a_cols[0]][col] += 1
        V[b_rows[0]*p + b_cols[1]][col] += 1
        V[b_rows[1]*p + b_cols[1]][col] -= 1
        # Z01 += M3, Z11 += M3
        W[c_cols[1]*n + c_rows[0]][col] += 1
        W[c_cols[1]*n + c_rows[1]][col] += 1
        
        # M4: X11*(Y10-Y00)
        col = base + 3
        U[a_rows[1]*m + a_cols[1]][col] += 1
        V[b_rows[1]*p + b_cols[0]][col] += 1
        V[b_rows[0]*p + b_cols[0]][col] -= 1
        # Z00 += M4, Z10 += M4
        W[c_cols[0]*n + c_rows[0]][col] += 1
        W[c_cols[0]*n + c_rows[1]][col] += 1
        
        # M5: (X00+X01)*Y11
        col = base + 4
        U[a_rows[0]*m + a_cols[0]][col] += 1
        U[a_rows[0]*m + a_cols[1]][col] += 1
        V[b_rows[1]*p + b_cols[1]][col] += 1
        # Z00 -= M5, Z01 += M5
        W[c_cols[0]*n + c_rows[0]][col] -= 1
        W[c_cols[1]*n + c_rows[0]][col] += 1
        
        # M6: (X10-X00)*(Y00+Y01)
        col = base + 5
        U[a_rows[1]*m + a_cols[0]][col] += 1
        U[a_rows[0]*m + a_cols[0]][col] -= 1
        V[b_rows[0]*p + b_cols[0]][col] += 1
        V[b_rows[0]*p + b_cols[1]][col] += 1
        # Z11 += M6
        W[c_cols[1]*n + c_rows[1]][col] += 1
        
        # M7: (X01-X11)*(Y10+Y11)
        col = base + 6
        U[a_rows[0]*m + a_cols[1]][col] += 1
        U[a_rows[1]*m + a_cols[1]][col] -= 1
        V[b_rows[1]*p + b_cols[0]][col] += 1
        V[b_rows[1]*p + b_cols[1]][col] += 1
        # Z00 += M7
        W[c_cols[0]*n + c_rows[0]][col] += 1
    
    # Block 0: A_L * B_TL -> C_L
    # A_L = A[0:2, 0:2], B_TL = B[0:2, 0:2], C_L = C[0:2, 0:2]
    add_strassen_block(0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 1: A_R * B_BL -> C_L
    # A_R = A[0:2, 2:4], B_BL = B[2:4, 0:2], C_L = C[0:2, 0:2]
    add_strassen_block(1, [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A_L * B_TR -> C_R
    # A_L = A[0:2, 0:2], B_TR = B[0:2, 2:4], C_R = C[0:2, 2:4]
    add_strassen_block(2, [0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Block 3: A_R * B_BR -> C_R
    # A_R = A[0:2, 2:4], B_BR = B[2:4, 2:4], C_R = C[0:2, 2:4]
    add_strassen_block(3, [0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    return {"U": U, "V": V, "W": W}
