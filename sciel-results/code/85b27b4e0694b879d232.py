"""Program for finding rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose the 2x3x8 matrix multiplication using dual numbers embedding.
    
    Strategy:
    - Embed B (3x8) into a 2x4 matrix over dual numbers R[epsilon]/(epsilon^2)
    - B_embed[i][j] = B[2*i][j] + epsilon * B[2*i+1][j] for i=0,1 and j=0..3
      But B has 3 rows, not 4. Let me reconsider.
    
    Actually, let's think differently. We have A (2x3) * B (3x8) = C (2x8).
    
    Let me split B into two 3x4 blocks: B_left (cols 0-3), B_right (cols 4-7)
    C = [C_left, C_right] where C_left = A*B_left, C_right = A*B_right
    
    For each 2x3x4 block, use dual numbers:
    Embed A (2x3) into 2x2 over dual numbers:
    A_embed[0][0] = A[0][0] + eps*A[0][1]
    A_embed[0][1] = A[0][2]
    A_embed[1][0] = A[1][0] + eps*A[1][1]  
    A_embed[1][1] = A[1][2]
    
    Embed B (3x4) into 2x4 over dual numbers:
    B_embed[0][j] = B[0][j] + eps*B[1][j]
    B_embed[1][j] = B[2][j]
    
    Then A_embed * B_embed gives a 2x4 matrix over dual numbers.
    The real part gives A[0:2,0:2]*B[0:2,:] + ... hmm this is getting complex.
    
    Let me use a direct construction with Strassen on 2x2x2 blocks.
    
    Split A (2x3) = [A1 (2x2), A2 (2x1)]
    Split B (3x8) = [[B1 (2x8)], [B2 (1x8)]]
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 x 2x8):
    Split B1 into four 2x2 blocks: B1_0, B1_1, B1_2, B1_3
    Each A1*B1_i is 2x2x2, use Strassen (rank 7 each)
    Total: 4*7 = 28
    
    For A2*B2 (2x1 x 1x8):
    This is 2*1*8 = 16 schoolbook multiplications
    
    Total: 28 + 16 = 44
    
    Let me implement this.
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8) using Strassen on four 2x2x2 blocks
    # A1 = A[0:2, 0:2], B1 = B[0:2, 0:8]
    # B1 split into B1_0 (cols 0-1), B1_1 (cols 2-3), B1_2 (cols 4-5), B1_3 (cols 6-7)
    
    # Strassen for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A10)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def add_strassen_2x2x2(a_row0, a_row1, b_col0, b_col1, b_col2, b_col3, start_col):
        """
        Add Strassen decomposition for 2x2 matrix A * 2x2 matrix B.
        A = [[A00, A01], [A10, A11]] where:
          A00 = A[a_row0][b_col0], A01 = A[a_row0][b_col1]
          A10 = A[a_row1][b_col0], A11 = A[a_row1][b_col1]
        B = [[B00, B01], [B10, B11]] where:
          B00 = B[b_col0][b_col0], B01 = B[b_col0][b_col1]
          B10 = B[b_col1][b_col0], B11 = B[b_col1][b_col1]
        C = A*B, C[i][j] contributes to W[(b_col0+2*j)*n + i] and W[(b_col1+2*j)*n + i]
        
        Returns new column index after adding 7 terms.
        """
        global U, V, W
        c = start_col
        
        # M1 = (A00+A11)*(B00+B11)
        U[a_row0][c] = 1; V[b_col0][c] = 1; W[(b_col0)*n + a_row0][c] = 1
        U[a_row1][c] = 1; V[b_col0][c] = 1; W[(b_col0)*n + a_row1][c] = 1
        U[a_row0][c] = 1; V[b_col1][c] = 1; W[(b_col1)*n + a_row0][c] = 1
        U[a_row1][c] = 1; V[b_col1][c] = 1; W[(b_col1)*n + a_row1][c] = 1
        c += 1
        
        # M2 = (A10+A11)*B00
        U[a_row1][c] = 1; V[b_col0][c] = 1; W[(b_col0)*n + a_row1][c] = 1
        U[a_row1][c] = 1; V[b_col0][c] = 1; W[(b_col0)*n + a_row1][c] = 1
        c += 1
        
        # M3 = A00*(B01-B11)
        U[a_row0][c] = 1; V[b_col0][c] = 1; W[(b_col1)*n + a_row0][c] = 1
        U[a_row0][c] = 1; V[b_col1][c] = 1; W[(b_col1)*n + a_row0][c] = -1
        c += 1
        
        # M4 = A11*(B10-B00)
        U[a_row1][c] = 1; V[b_col1][c] = 1; W[(b_col0)*n + a_row1][c] = 1
        U[a_row1][c] = 1; V[b_col0][c] = 1; W[(b_col0)*n + a_row1][c] = -1
        c += 1
        
        # M5 = (A00+A01)*B11
        U[a_row0][c] = 1; V[b_col1][c] = 1; W[(b_col1)*n + a_row0][c] = 1
        U[a_row0][c] = 1; V[b_col1][c] = 1; W[(b_col1)*n + a_row0][c] = 1
        c += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[a_row1][c] = 1; V[b_col0][c] = 1; W[(b_col0)*n + a_row1][c] = 1
        U[a_row0][c] = 1; V[b_col0][c] = 1; W[(b_col0)*n + a_row0][c] = -1
        U[a_row1][c] = 1; V[b_col1][c] = 1; W[(b_col1)*n + a_row1][c] = 1
        U[a_row0][c] = 1; V[b_col1][c] = 1; W[(b_col1)*n + a_row0][c] = -1
        c += 1
        
        # M7 = (A01-A10)*(B10+B11)
        U[a_row0][c] = 1; V[b_col1][c] = 1; W[(b_col0)*n + a_row0][c] = 1
        U[a_row1][c] = 1; V[b_col1][c] = 1; W[(b_col0)*n + a_row1][c] = -1
        U[a_row0][c] = 1; V[b_col1][c] = 1; W[(b_col1)*n + a_row0][c] = 1
        U[a_row1][c] = 1; V[b_col1][c] = 1; W[(b_col1)*n + a_row1][c] = -1
        c += 1
        
        return c
    
    # This approach is getting too complex. Let me use a simpler schoolbook-based
    # decomposition that's guaranteed correct, then optimize.
    
    # Reset to schoolbook for now to ensure correctness
    rank = n * m * p  # 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
