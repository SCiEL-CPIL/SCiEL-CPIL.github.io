import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    DEADLINE = start_time + 135  # safety margin before 150s kill
    
    # Build the rank-56 Strassen decomposition
    rank_56 = 56
    U = np.zeros((8, rank_56))
    V = np.zeros((32, rank_56))
    W = np.zeros((16, rank_56))
    
    col = 0
    # We have 4 blocks of 2x2x2 along the p dimension (8 = 4*2)
    # And 1 block of 2x2x2 along the m dimension? No.
    # A is 2x4. B is 4x8.
    # Split A into two 2x2 blocks horizontally: A_L, A_R.
    # Split B into four 2x2 blocks vertically?
    # B is 4x8.
    # Let's split B into 4 columns of width 2.
    # B = [B0, B1, B2, B3] where each Bi is 4x2.
    # Each Bi can be split into two 2x2 blocks vertically: Bi = [Bi_top; Bi_bottom].
    # So we have 8 blocks of 2x2x2.
    # For each block k (0..7):
    #   A_block is either A_L (cols 0-1) or A_R (cols 2-3).
    #   B_block is the k-th 2x2 submatrix of B.
    #   C_block is the k-th 2x2 submatrix of C.
    
    # A is 2x4.
    # A_L = A[:, 0:2]
    # A_R = A[:, 2:4]
    
    # B is 4x8.
    # B_0 = B[0:2, 0:2] -> uses A_L
    # B_1 = B[2:4, 0:2] -> uses A_L
    # B_2 = B[0:2, 2:4] -> uses A_R
    # B_3 = B[2:4, 2:4] -> uses A_R
    # B_4 = B[0:2, 4:6] -> uses A_L
    # B_5 = B[2:4, 4:6] -> uses A_L
    # B_6 = B[0:2, 6:8] -> uses A_R
    # B_7 = B[2:4, 6:8] -> uses A_R
    
    # So blocks 0,1,4,5 use A_L.
    # Blocks 2,3,6,7 use A_R.
    
    # Indices for A (2x4):
    # A_L: (0,0), (0,1), (1,0), (1,1)
    # A_R: (0,2), (0,3), (1,2), (1,3)
    
    # Indices for B (4x8):
    # Block 0: rows 0-1, cols 0-1
    # Block 1: rows 2-3, cols 0-1
    # Block 2: rows 0-1, cols 2-3
    # Block 3: rows 2-3, cols 2-3
    # Block 4: rows 0-1, cols 4-5
    # Block 5: rows 2-3, cols 4-5
    # Block 6: rows 0-1, cols 6-7
    # Block 7: rows 2-3, cols 6-7
    
    # Indices for C (2x8):
    # Block 0: cols 0-1
    # Block 1: cols 0-1 (wait, C is 2x8. The output of block 1 adds to C[:, 0:2]?)
    # No, C = A_L * B[:, 0:2] + A_R * B[:, 2:4] ...
    # C[:, 0:2] = A_L * B[0:2, 0:2] + A_L * B[2:4, 0:2] ?
    # No.
    # C = A * B.
    # C[:, j] = sum_k A[:, k] B[k, j].
    # If we split k into 0-1 and 2-3.
    # C[:, j] = A[:, 0:2] @ B[0:2, j] + A[:, 2:4] @ B[2:4, j].
    
    # So for output column pair j, j+1:
    # C[:, j:j+2] = A_L * B[0:2, j:j+2] + A_R * B[2:4, j:j+2].
    
    # So for each output block (4 of them, each 2x2):
    # We need to compute A_L * B_top and A_R * B_bottom.
    # Each is a 2x2 * 2x2 multiplication.
    # So we have 4 output blocks * 2 multiplications per block = 8 multiplications.
    # Each multiplication is rank 7.
    # Total rank 56.
    
    # Let's map the 8 multiplications to the 7 Strassen terms.
    
    # Multiplication 0: A_L * B[0:2, 0:2] -> C[0:2, 0:2]
    # Multiplication 1: A_R * B[2:4, 0:2] -> C[0:2, 0:2]
    # Multiplication 2: A_L * B[0:2, 2:4] -> C[0:2, 2:4]
    # Multiplication 3: A_R * B[2:4, 2:4] -> C[0:2, 2:4]
    # Multiplication 4: A_L * B[0:2, 4:6] -> C[0:2, 4:6]
    # Multiplication 5: A_R * B[2:4, 4:6] -> C[0:2, 4:6]
    # Multiplication 6: A_L * B[0:2, 6:8] -> C[0:2, 6:8]
    # Multiplication 7: A_R * B[2:4, 6:8] -> C[0:2, 6:8]
    
    # Strassen terms for A * B:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 - P2 - P4 + P7
    # C01 = P3 + P5
    # C10 = P2 + P6
    # C11 = P1 - P3 - P5 + P6
    
    # We will iterate through the 8 multiplications.
    # For each, we define the A indices and B indices.
    
    mults = [
        # (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
        # a_col_offset: 0 for A_L, 2 for A_R
        # b_row_offset: 0 for top, 2 for bottom
        # b_col_offset: 0, 2, 4, 6
        # c_col_offset: 0, 2, 4, 6
        
        # Mult 0: A_L * B[0:2, 0:2] -> C[0:2, 0:2]
        (0, 0, 0, 0),
        # Mult 1: A_R * B[2:4, 0:2] -> C[0:2, 0:2]
        (2, 2, 0, 0),
        
        # Mult 2: A_L * B[0:2, 2:4] -> C[0:2, 2:4]
        (0, 0, 2, 2),
        # Mult 3: A_R * B[2:4, 2:4] -> C[0:2, 2:4]
        (2, 2, 2, 2),
        
        # Mult 4: A_L * B[0:2, 4:6] -> C[0:2, 4:6]
        (0, 0, 4, 4),
        # Mult 5: A_R * B[2:4, 4:6] -> C[0:2, 4:6]
        (2, 2, 4, 4),
        
        # Mult 6: A_L * B[0:2, 6:8] -> C[0:2, 6:8]
        (0, 0, 6, 6),
        # Mult 7: A_R * B[2:4, 6:8] -> C[0:2, 6:8]
        (2, 2, 6, 6),
    ]
    
    for (a_col_off, b_row_off, b_col_off, c_col_off) in mults:
        # A indices
        a00 = 0*4 + a_col_off
        a01 = 0*4 + a_col_off + 1
        a10 = 1*4 + a_col_off
        a11 = 1*4 + a_col_off + 1
        
        # B indices
        b00 = b_row_off*8 + b_col_off
        b01 = b_row_off*8 + b_col_off + 1
        b10 = (b_row_off+1)*8 + b_col_off
        b11 = (b_row_off+1)*8 + b_col_off + 1
        
        # C indices
        c00 = c_col_off*2 + 0
        c01 = (c_col_off+1)*2 + 0
        c10 = c_col_off*2 + 1
        c11 = (c_col_off+1)*2 + 1
        
        # P1 = (a00+a11)(b00+b11)
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P2 = (a10+a11)*b00
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        W[c11, col] += -1
        col += 1
        
        # P3 = a00*(b01-b11)
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P4 = a11*(b10-b00)
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # P5 = (a00+a01)*b11
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1
        W[c01, col] += 1
        col += 1
        
        # P6 = (a10-a00)*(b00+b01)
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P7 = (a01-a11)*(b10+b11)
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1
        
    assert col == 56
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
