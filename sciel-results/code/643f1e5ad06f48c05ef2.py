import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0 # Safety margin
    
    n, m, p = 3, 4, 4
    
    # Local 3x2x2 decomposition (Rank 11)
    # A: 3x2, B: 2x2, C: 3x2
    # Indices: A(r,c), B(r,c), C(r,c)
    
    U_loc = np.zeros((6, 11), dtype=int)
    V_loc = np.zeros((4, 11), dtype=int)
    W_loc = np.zeros((6, 11), dtype=int)
    
    # Strassen 2x2x2 (Terms 0-6)
    # S1 = (a00+a11)(b00+b11) -> c00, c11
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # S2 = (a10+a11)b00 -> c10, c11
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    # S3 = a00(b10-b00) -> c01, c11
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    # S4 = a11(b01-b11) -> c00, c10
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # S5 = (a00+a01)b11 -> c00, c01
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # S6 = (a10-a00)(b00+b10) -> c11
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    # S7 = (a01-a11)(b01+b11) -> c00
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Direct terms for row 2 (Terms 7-10)
    # D1 = a20*b00 -> c20
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    # D2 = a21*b10 -> c20
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    # D3 = a20*b01 -> c21
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    # D4 = a21*b11 -> c21
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    # Global tensors
    U = np.zeros((n * m, 44), dtype=int)
    V = np.zeros((m * p, 44), dtype=int)
    W = np.zeros((p * n, 44), dtype=int)
    
    # Blocks: (a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset)
    # A is 3x4. B is 4x4. C is 4x3? No, C is 3x4?
    # Task: "output factor follows the official trace convention W[k*n+i,r]"
    # Matrix mult: A (3x4) * B (4x4) = C (3x4).
    # W shape is (p*n, rank) = (4*3, 44) = (12, 44).
    # Index in W: k*n + i. k is row of C (0..3), i is col of C (0..2)?
    # Wait. C is 3x4. Rows 0..2, Cols 0..3.
    # If W is (p*n, rank), and p=4, n=3.
    # Usually W[i, j] corresponds to C[j, i]? Or C[i, j]?
    # "W[k*n+i,r]".
    # If k is the column index of B (which is row index of C?) and i is row index of A (col index of C?)
    # Let's check the seed's logic or standard conventions.
    # In the current code:
    # W[gi, ...] where gi = (cc + c) * 3 + (cr + r).
    # Here cc is c_col_offset (0 or 2). c is 0 or 1.
    # So first index of W is (col_C) * 3 + (row_C).
    # This implies W is stored in column-major order of C?
    # C has rows 0..2, cols 0..3.
    # Index = col * n + row.
    # This matches "W[k*n+i,r]" if k=col, i=row.
    
    blocks = [
        (0, 0, 0, 0, 0), # Block 0: A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
        (2, 2, 0, 0, 0), # Block 1: A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
        (0, 0, 2, 0, 2), # Block 2: A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
        (2, 2, 2, 0, 2)  # Block 3: A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
    ]
    
    for bi, (ac, br, bc, cr, cc) in enumerate(blocks):
        off = bi * 11
        
        # Map U (A factors)
        # A global index: r * m + c = r * 4 + c
        for r in range(3):
            for c in range(2):
                li = r * 2 + c
                gi = r * 4 + (ac + c)
                U[gi, off:off+11] += U_loc[li, :]
                
        # Map V (B factors)
        # B global index: (br + r) * p + (bc + c) = (br+r)*4 + (bc+c)
        for r in range(2):
            for c in range(2):
                li = r * 2 + c
                gi = (br + r) * 4 + (bc + c)
                V[gi, off:off+11] += V_loc[li, :]
                
        # Map W (C factors)
        # C global index: (cc + c) * n + (cr + r) = (cc+c)*3 + (cr+r)
        # Note: cr is always 0 in our blocks because C rows are 0..2 and we cover all rows.
        for r in range(3):
            for c in range(2):
                li = r * 2 + c
                gi = (cc + c) * 3 + (cr + r)
                W[gi, off:off+11] += W_loc[li, :]
                
    if time.time() > DEADLINE:
        # Should not happen
        pass
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
