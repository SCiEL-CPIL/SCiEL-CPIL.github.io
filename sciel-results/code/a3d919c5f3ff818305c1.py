"""Matrix multiplication 2x3x7 with rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split m = 3 into 1 + 2.
    # A is 2x3, B is 3x7.
    # C = A * B = A[:,0]*B[0,:] + A[:,1:3]*B[1:3,:]
    
    # Part 1: A1 * B1 where A1 is 2x1, B1 is 1x7.
    # This is essentially 2 outer products of size 1x7 each.
    # Rank = 2 * 7 = 14.
    # Terms: for i in 0..1, k in 0..6: A[i,0] * B[0,k]
    
    # Part 2: A2 * B2 where A2 is 2x2, B2 is 2x7.
    # Split B2 columns into three blocks of 2 (for Strassen) and one block of 1 (for schoolbook).
    # B2 = [B2_0, B2_1, B2_2, B2_3] where B2_0, B2_1, B2_2 are 2x2, B2_3 is 2x1.
    # C2 = A2 * B2_0 + A2 * B2_1 + A2 * B2_2 + A2 * B2_3
    # Each A2 * B2_j for j=0,1,2 is a 2x2x2 multiplication -> Strassen rank 7.
    # A2 * B2_3 is a 2x2x1 multiplication -> schoolbook rank 4.
    # Total rank = 14 + 3*7 + 4 = 39.
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # 6 x 39
    V = [[0] * rank for _ in range(m * p)]  # 21 x 39
    W = [[0] * rank for _ in range(p * n)]  # 14 x 39
    
    col = 0
    
    # Part 1: A1 * B1 (14 terms)
    # A1 = A[:, 0], B1 = B[0, :]
    # Term (i, k): A[i, 0] * B[0, k]
    # U index: i * m + 0 = i * 3
    # V index: 0 * p + k = k
    # W index: k * n + i = k * 2 + i
    for i in range(n):  # 0..1
        for k in range(p):  # 0..6
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col = 14
    
    # Strassen for 2x2x2:
    # Given matrices X (2x2) and Y (2x2), compute Z = X * Y.
    # Strassen's 7 products:
    # M1 = (X00 + X11) * (Y00 + Y11)
    # M2 = (X10 + X11) * Y00
    # M3 = X00 * (Y01 - Y11)
    # M4 = X11 * (Y10 - Y00)
    # M5 = (X00 + X01) * Y11
    # M6 = (X10 - X00) * (Y00 + Y01)
    # M7 = (X01 - X11) * (Y10 + Y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For A2 * B2_j:
    # A2 is rows 1,2 of A (indices 1,2 in column 1..2 of A)
    # A2[0,0] = A[1,1], A2[0,1] = A[1,2]
    # A2[1,0] = A[2,1], A2[1,1] = A[2,2]
    # B2_j is columns 2j, 2j+1 of B[1:3, :]
    # B2_j[0,0] = B[1, 2j], B2_j[0,1] = B[1, 2j+1]
    # B2_j[1,0] = B[2, 2j], B2_j[1,1] = B[2, 2j+1]
    
    # Strassen decomposition for one 2x2x2 block:
    def strassen_block(U, V, W, col, row_offset, col_offset):
        # row_offset: starting row index in A (0-based within the 2x2 block)
        # col_offset: starting column index in B (0-based within the 2x2 block)
        # For A2, rows are global indices 1, 2 (since A[:, 1:3])
        # For B2_j, columns are global indices 2j, 2j+1 (since B[1:3, :])
        
        # A2 row 0 = global row 1, A2 row 1 = global row 2
        # B2_j col 0 = global col 2j, B2_j col 1 = global col 2j+1
        
        # U indices for A2:
        # A2[0,0] -> U[1*3 + 1] = U[4]
        # A2[0,1] -> U[1*3 + 2] = U[5]
        # A2[1,0] -> U[2*3 + 1] = U[7]
        # A2[1,1] -> U[2*3 + 2] = U[8]
        
        # V indices for B2_j:
        # B2_j[0,0] -> V[1*7 + 2j] = V[7 + 2j]
        # B2_j[0,1] -> V[1*7 + 2j+1] = V[7 + 2j + 1]
        # B2_j[1,0] -> V[2*7 + 2j] = V[14 + 2j]
        # B2_j[1,1] -> V[2*7 + 2j+1] = V[14 + 2j + 1]
        
        a00 = 1 * m + 1  # 4
        a01 = 1 * m + 2  # 5
        a10 = 2 * m + 1  # 7
        a11 = 2 * m + 2  # 8
        
        b00 = 1 * p + 2 * col_offset      # 7 + 2*col_offset
        b01 = 1 * p + 2 * col_offset + 1  # 7 + 2*col_offset + 1
        b10 = 2 * p + 2 * col_offset      # 14 + 2*col_offset
        b11 = 2 * p + 2 * col_offset + 1  # 14 + 2*col_offset + 1
        
        # W indices for output C2_j (2x2):
        # C2_j[0,0] -> W[2j * n + 0] = W[4j]
        # C2_j[0,1] -> W[(2j+1) * n + 0] = W[4j + 2]
        # C2_j[1,0] -> W[2j * n + 1] = W[4j + 1]
        # C2_j[1,1] -> W[(2j+1) * n + 1] = W[4j + 3]
        
        w00 = (2 * col_offset) * n + 0      # 4j
        w01 = (2 * col_offset + 1) * n + 0  # 4j + 2
        w10 = (2 * col_offset) * n + 1      # 4j + 1
        w11 = (2 * col_offset + 1) * n + 1  # 4j + 3
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M2 = (a10 + a11) * b00
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[w10][col] += 1
        W[w11][col] -= 1
        col += 1
        
        # M3 = a00 * (b01 - b11)
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M4 = a11 * (b10 - b00)
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # M5 = (a00 + a01) * b11
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[w00][col] -= 1
        W[w01][col] += 1
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[w00][col] += 1
        col += 1
        
        return col
    
    # Three Strassen blocks for columns 0-1, 2-3, 4-5 of B2
    for j in range(3):
        col = strassen_block(U, V, W, col, 0, j)
    # col = 14 + 21 = 35
    
    # Part 2, block 3: A2 * B2_3 where B2_3 is 2x1 (column 6 of B, rows 1,2)
    # Schoolbook: 4 terms
    # A2[0,0]*B2_3[0,0] -> U[4]*V[7+6]*W[6*2+0] = U[4]*V[13]*W[12]
    # A2[0,1]*B2_3[1,0] -> U[5]*V[14+6]*W[6*2+0] = U[5]*V[20]*W[12]
    # A2[1,0]*B2_3[0,0] -> U[7]*V[13]*W[6*2+1] = U[7]*V[13]*W[13]
    # A2[1,1]*B2_3[1,0] -> U[8]*V[20]*W[6*2+1] = U[8]*V[20]*W[13]
    
    # B2_3 is column 6 of B, rows 1,2.
    # B2_3[0,0] = B[1,6] -> V[1*7+6] = V[13]
    # B2_3[1,0] = B[2,6] -> V[2*7+6] = V[20]
    
    # Output C2_3 is column 6 of C, rows 0,1.
    # C2_3[0,0] = C[0,6] -> W[6*2+0] = W[12]
    # C2_3[1,0] = C[1,6] -> W[6*2+1] = W[13]
    
    # Term 1: A2[0,0] * B2_3[0,0]
    U[4][col] = 1
    V[13][col] = 1
    W[12][col] = 1
    col += 1
    
    # Term 2: A2[0,1] * B2_3[1,0]
    U[5][col] = 1
    V[20][col] = 1
    W[12][col] = 1
    col += 1
    
    # Term 3: A2[1,0] * B2_3[0,0]
    U[7][col] = 1
    V[13][col] = 1
    W[13][col] = 1
    col += 1
    
    # Term 4: A2[1,1] * B2_3[1,0]
    U[8][col] = 1
    V[20][col] = 1
    W[13][col] = 1
    col += 1
    
    # col should be 39
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
