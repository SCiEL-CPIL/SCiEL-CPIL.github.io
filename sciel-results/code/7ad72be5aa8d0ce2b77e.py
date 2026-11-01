import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 2x2 * 2x14 matrix multiplication.
    
    Uses Strassen's algorithm for each of the 7 independent 2x2 blocks.
    Total rank = 49.
    
    The 2x2 matrix A multiplies a 2x14 matrix B to produce a 14x2 matrix C.
    We view B as 7 independent 2x2 blocks (columns 2r, 2r+1 for r=0..6).
    For each block, we use Strassen's 7-multiplication algorithm.
    
    Conventions:
      U: (n*m, rank) = (4, 49)
      V: (m*p, rank) = (28, 49)
      W: (p*n, rank) = (28, 49)
      W index: k*n + i  (row k of C, column i)
      V index: j*p + k  (row j of B, column k)
      U index: i*m + j  (row i of A, column j)
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7  # 49
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen linear forms for A (2x2)
    # A flat indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    # Strassen products:
    #   M1 = (A00 + A11) * (B00 + B11)
    #   M2 = (A10 + A11) * B00
    #   M3 = A00 * (B01 - B11)
    #   M4 = A11 * (B10 - B00)
    #   M5 = (A00 + A01) * B11
    #   M6 = (A10 - A00) * (B00 + B01)
    #   M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_forms = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1]   # M7: A01 - A11
    ]
    
    # V forms for each Strassen product (B flat indices within a 2x2 block)
    # B flat indices in block: 0=B00, 1=B01, 2=B10, 3=B11
    V_forms = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [-1, 0, 1, 0],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1]    # M7: B10 + B11
    ]
    
    # W forms for each Strassen product (C flat indices within a 2x2 block)
    # C flat indices in block: 0=C00, 1=C01, 2=C10, 3=C11
    W_forms = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, -1],  # M2: C10 - C11
        [0, 1, 0, 1],   # M3: C01 + C11
        [1, 0, 1, 0],   # M4: C00 + C10
        [-1, 1, 0, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0]    # M7: C00
    ]
    
    for r in range(num_blocks):
        # B block r: columns 2r, 2r+1 of B (2x14)
        # B flat index: j*p + k
        b_base = 0  # row 0 of B
        b_row0_col0 = 0 * p + 2 * r       # B[0, 2r]
        b_row0_col1 = 0 * p + 2 * r + 1   # B[0, 2r+1]
        b_row1_col0 = 1 * p + 2 * r       # B[1, 2r]
        b_row1_col1 = 1 * p + 2 * r + 1   # B[1, 2r+1]
        
        # C block r: rows 2r, 2r+1 of C (14x2)
        # C flat index: k*n + i
        c_row0_col0 = 2 * r * n + 0       # C[2r, 0]
        c_row0_col1 = 2 * r * n + 1       # C[2r, 1]
        c_row1_col0 = (2 * r + 1) * n + 0 # C[2r+1, 0]
        c_row1_col1 = (2 * r + 1) * n + 1 # C[2r+1, 1]
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U
            for i in range(4):
                U[i][c] = float(U_forms[s][i])
            
            # Fill V
            V[b_row0_col0][c] = float(V_forms[s][0])
            V[b_row0_col1][c] = float(V_forms[s][1])
            V[b_row1_col0][c] = float(V_forms[s][2])
            V[b_row1_col1][c] = float(V_forms[s][3])
            
            # Fill W
            W[c_row0_col0][c] = float(W_forms[s][0])
            W[c_row0_col1][c] = float(W_forms[s][1])
            W[c_row1_col0][c] = float(W_forms[s][2])
            W[c_row1_col1][c] = float(W_forms[s][3])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
