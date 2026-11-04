"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 2x4 by 4x5.
    # Split the 4 middle dimension into two groups of 2.
    # Each 2x2x5 sub-problem is further decomposed using Strassen on the 2x2 blocks
    # with schoolbook for the remaining dimension.
    #
    # For a 2x2 matrix A and 2x2 matrix B, Strassen computes C = A*B with 7 multiplications:
    #   M1 = (A11 + A22) * (B11 + B22)
    #   M2 = (A21 + A22) * B11
    #   M3 = A11 * (B12 - B22)
    #   M4 = A22 * (B21 - B11)
    #   M5 = (A11 + A12) * B22
    #   M6 = (A21 - A11) * (B11 + B12)
    #   M7 = (A12 - A22) * (B21 + B22)
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #
    # For 2x2 by 2x5: we have A (2x2) and B (2x5).
    # We can treat B as two 2x2 blocks B_left (cols 0,1) and B_right (cols 2,3,4)... 
    # Actually, let's think differently.
    #
    # 2x4 by 4x5: A is 2x4, B is 4x5.
    # Split A into A_left (2x2, cols 0,1) and A_right (2x2, cols 2,3).
    # Split B into B_top (2x5, rows 0,1) and B_bottom (2x5, rows 2,3).
    # C = A_left * B_top + A_right * B_bottom
    #
    # Each term is 2x2 by 2x5. Let's decompose 2x2 by 2x5.
    # A (2x2), B (2x5). 
    # We can use Strassen-like approach on the 2x2 part of A and the 2x2 part of B.
    # But B has 5 columns. Let's split B into B1 (2x2, cols 0,1) and B2 (2x2, cols 2,3) and B3 (2x1, col 4).
    # C = A*B1 + A*B2 + A*B3
    # Each A*B_i for i=1,2 is 2x2 by 2x2, which uses Strassen (rank 7).
    # A*B3 is 2x2 by 2x1, which is schoolbook (rank 4).
    # So 2x2 by 2x5 has rank 7+7+4 = 18.
    #
    # Total rank = 18 + 18 = 36.
    #
    # Let's construct this explicitly.
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    # For each of the two 2x2x5 sub-problems (left and right halves of the middle dimension)
    for half in range(2):
        # A_half is rows 0,1 of A (which are all rows since n=2), cols 2*half, 2*half+1
        # B_half is rows 2*half, 2*half+1 of B, all 5 cols
        # 
        # Decompose A_half (2x2) by B_half (2x5) into Strassen parts
        # A_half = [[a00, a01], [a10, a11]] where a00 = A[0][2*half], a01 = A[0][2*half+1], etc.
        # B_half = [[b00, b01, b02, b03, b04], [b10, b11, b12, b13, b14]]
        
        # Split B_half into B1 (cols 0,1), B2 (cols 2,3), B3 (col 4)
        
        # Strassen for A_half * B1 (2x2 by 2x2)
        # M1 = (a00+a11)*(b00+b10)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        
        # For M1: U contribution from a00 and a11, V from b00 and b10, W from c00
        # c00 gets M1, M4, -M5, M7
        # c01 gets M3, M5
        # c10 gets M2, M4
        # c11 gets M1, -M2, M3, M6
        
        # Let's build the U, V, W matrices for the Strassen part.
        # U row indices: 0,1 for A_half rows, mapped to global A rows 0,1
        # But A_half cols are 2*half, 2*half+1, so global A col indices are 2*half, 2*half+1
        # U is indexed by [i*m + j] where i is row, j is col of A.
        # So A_half[0][0] = U[0*m + 2*half] = U[2*half]
        # A_half[0][1] = U[0*m + 2*half+1] = U[2*half+1]
        # A_half[1][0] = U[1*m + 2*half] = U[m + 2*half] = U[4 + 2*half]
        # A_half[1][1] = U[1*m + 2*half+1] = U[m + 2*half+1] = U[4 + 2*half+1]
        
        # V is indexed by [j*p + k] where j is row of B, k is col of B.
        # B_half rows are 2*half, 2*half+1, cols 0..4
        # B1 cols are 0,1 of B_half, so global B cols are 0,1
        # B_half[0][0] = V[2*half*p + 0] = V[10*half]
        # B_half[0][1] = V[2*half*p + 1] = V[10*half+1]
        # B_half[1][0] = V[(2*half+1)*p + 0] = V[10*half+10]
        # B_half[1][1] = V[(2*half+1)*p + 1] = V[10*half+11]
        
        # W is indexed by [k*n + i] where k is col of C, i is row of C.
        # C rows are 0,1, cols are 0..4
        # For B1, C cols are 0,1
        # C[0][0] = W[0*2+0] = W[0]
        # C[0][1] = W[1*2+0] = W[2]
        # C[1][0] = W[0*2+1] = W[1]
        # C[1][1] = W[1*2+1] = W[3]
        
        # For B2, C cols are 2,3
        # C[0][2] = W[2*2+0] = W[4]
        # C[0][3] = W[3*2+0] = W[6]
        # C[1][2] = W[2*2+1] = W[5]
        # C[1][3] = W[3*2+1] = W[7]
        
        # For B3, C col is 4
        # C[0][4] = W[4*2+0] = W[8]
        # C[1][4] = W[4*2+1] = W[9]
        
        # Strassen for A_half * B1 (cols 0,1 of B_half)
        # a00 = U[2*half], a01 = U[2*half+1], a10 = U[4+2*half], a11 = U[4+2*half+1]
        # b00 = V[10*half], b01 = V[10*half+1], b10 = V[10*half+10], b11 = V[10*half+11]
        
        # M1 = (a00+a11)*(b00+b10)
        U[2*half][col] = 1
        U[4+2*half+1][col] = 1
        V[10*half][col] = 1
        V[10*half+10][col] = 1
        W[0][col] = 1   # c00
        W[3][col] = 1   # c11
        col += 1
        
        # M2 = (a10+a11)*b00
        U[4+2*half][col] = 1
        U[4+2*half+1][col] = 1
        V[10*half][col] = 1
        W[1][col] = 1   # c10
        W[3][col] = -1  # c11
        col += 1
        
        # M3 = a00*(b01-b11)
        U[2*half][col] = 1
        V[10*half+1][col] = 1
        V[10*half+11][col] = -1
        W[2][col] = 1   # c01
        W[3][col] = 1   # c11
        col += 1
        
        # M4 = a11*(b10-b00)
        U[4+2*half+1][col] = 1
        V[10*half+10][col] = 1
        V[10*half][col] = -1
        W[0][col] = 1   # c00
        W[1][col] = 1   # c10
        col += 1
        
        # M5 = (a00+a01)*b11
        U[2*half][col] = 1
        U[2*half+1][col] = 1
        V[10*half+11][col] = 1
        W[0][col] = -1  # c00
        W[2][col] = 1   # c01
        col += 1
        
        # M6 = (a10-a00)*(b00+b01)
        U[4+2*half][col] = 1
        U[2*half][col] = -1
        V[10*half][col] = 1
        V[10*half+1][col] = 1
        W[3][col] = 1   # c11
        col += 1
        
        # M7 = (a01-a11)*(b10+b11)
        U[2*half+1][col] = 1
        U[4+2*half+1][col] = -1
        V[10*half+10][col] = 1
        V[10*half+11][col] = 1
        W[0][col] = 1   # c00
        col += 1
        
        # Strassen for A_half * B2 (cols 2,3 of B_half)
        # b00 = V[10*half+2], b01 = V[10*half+3], b10 = V[10*half+12], b11 = V[10*half+13]
        # C cols 2,3: c02=W[4], c03=W[6], c12=W[5], c13=W[7]
        
        # M1 = (a00+a11)*(b00+b10)
        U[2*half][col] = 1
        U[4+2*half+1][col] = 1
        V[10*half+2][col] = 1
        V[10*half+12][col] = 1
        W[4][col] = 1   # c02
        W[7][col] = 1   # c13
        col += 1
        
        # M2 = (a10+a11)*b00
        U[4+2*half][col] = 1
        U[4+2*half+1][col] = 1
        V[10*half+2][col] = 1
        W[5][col] = 1   # c12
        W[7][col] = -1  # c13
        col += 1
        
        # M3 = a00*(b01-b11)
        U[2*half][col] = 1
        V[10*half+3][col] = 1
        V[10*half+13][col] = -1
        W[6][col] = 1   # c03
        W[7][col] = 1   # c13
        col += 1
        
        # M4 = a11*(b10-b00)
        U[4+2*half+1][col] = 1
        V[10*half+12][col] = 1
        V[10*half+2][col] = -1
        W[4][col] = 1   # c02
        W[5][col] = 1   # c12
        col += 1
        
        # M5 = (a00+a01)*b11
        U[2*half][col] = 1
        U[2*half+1][col] = 1
        V[10*half+13][col] = 1
        W[4][col] = -1  # c02
        W[6][col] = 1   # c03
        col += 1
        
        # M6 = (a10-a00)*(b00+b01)
        U[4+2*half][col] = 1
        U[2*half][col] = -1
        V[10*half+2][col] = 1
        V[10*half+3][col] = 1
        W[7][col] = 1   # c13
        col += 1
        
        # M7 = (a01-a11)*(b10+b11)
        U[2*half+1][col] = 1
        U[4+2*half+1][col] = -1
        V[10*half+12][col] = 1
        V[10*half+13][col] = 1
        W[4][col] = 1   # c02
        col += 1
        
        # Schoolbook for A_half * B3 (col 4 of B_half)
        # a00*b00, a01*b10, a10*b00, a11*b10
        # b00 = V[10*half+4], b10 = V[10*half+14]
        # C cols 4: c04=W[8], c14=W[9]
        
        # a00*b00 -> c04
        U[2*half][col] = 1
        V[10*half+4][col] = 1
        W[8][col] = 1
        col += 1
        
        # a01*b10 -> c04
        U[2*half+1][col] = 1
        V[10*half+14][col] = 1
        W[8][col] = 1
        col += 1
        
        # a10*b00 -> c14
        U[4+2*half][col] = 1
        V[10*half+4][col] = 1
        W[9][col] = 1
        col += 1
        
        # a11*b10 -> c14
        U[4+2*half+1][col] = 1
        V[10*half+14][col] = 1
        W[9][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
