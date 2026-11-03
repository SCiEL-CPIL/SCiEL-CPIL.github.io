import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will use Strassen for the 4x4x4 part and schoolbook for the 5th column.
    # Rank 49 for 4x4x4, Rank 16 for 4x4x1. Total 65.
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for 4x4x4
    # We define the 7 block multiplications.
    # Each block multiplication is a 2x2x2 Strassen.
    
    # Indices for 4x4 matrix A: i in 0..3, j in 0..3.
    # Block indices: r = i//2, c = j//2. Local: u = i%2, v = j%2.
    # A[i, j] corresponds to A_block[r, c][u, v]
    
    # The 7 Strassen combinations for 2x2 block multiplication:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)(B00)
    # M3 = (A00)(B10 - B11)
    # M4 = (A11)(B01 - B00)
    # M5 = (A00 + A10)(B11)
    # M6 = (A11 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B11 + B01)
    
    # For each block multiplication s (0..6), we have 7 scalar multiplications t (0..6).
    # The scalar multiplications come from the Strassen algorithm for 2x2 matrices.
    
    # Let's define the Strassen 2x2 factors.
    # For matrices X and Y (2x2), the 7 products are:
    # P1 = (X00 + X11)(Y00 + Y11)
    # P2 = (X10 + X11)(Y00)
    # P3 = (X00)(Y10 - Y11)
    # P4 = (X11)(Y01 - Y00)
    # P5 = (X00 + X10)(Y11)
    # P6 = (X11 - X00)(Y00 + Y10)
    # P7 = (X01 - X11)(Y11 + Y01)
    
    # The result Z = XY is:
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # We need to map the block indices to the actual matrix indices.
    
    # For a given block multiplication s, X is a 2x2 matrix formed from blocks of A.
    # Y is a 2x2 matrix formed from blocks of B.
    # X_rc is a 2x2 matrix (block of A).
    # Y_cd is a 2x2 matrix (block of B).
    
    # Let's define the coefficients for X and Y for each s.
    # X is 2x2, Y is 2x2.
    # X = [[X00, X01], [X10, X11]]
    # Y = [[Y00, Y01], [Y10, Y11]]
    
    # For s=0 (M1):
    # X00 = A00, X01 = 0, X10 = 0, X11 = A11  -> Wait, X is the matrix A00+A11?
    # No, in Strassen, we form the matrices first.
    # For M1, we multiply (A00 + A11) by (B00 + B11).
    # So X = A00 + A11. Since A00 and A11 are 2x2 blocks, X is a 2x2 matrix where:
    # X[0,0] = A[0,0] + A[2,2]
    # X[0,1] = A[0,1] + A[2,3]
    # X[1,0] = A[1,0] + A[3,2]
    # X[1,1] = A[1,1] + A[3,3]
    
    # Similarly for Y.
    
    # Let's create a helper to add a Strassen 2x2x2 decomposition.
    
    def add_strassen_2x2(X_coeffs, Y_coeffs, W_coeffs, start_col):
        """
        X_coeffs: list of 4 lists of 4 tuples (i, j, coeff) for the 2x2 matrix X.
                  X_coeffs[r][c] is a list of (i, j, coeff) for the block X_rc.
                  Actually, X is a 2x2 matrix of scalars in the context of the inner Strassen?
                  No.
                  Let's clarify the levels.
                  
                  Level 1: 4x4x4 multiplication.
                  We break A and B into 2x2 blocks of 2x2 matrices.
                  We perform 7 block multiplications.
                  Each block multiplication is a 2x2x2 multiplication.
                  
                  Level 2: 2x2x2 multiplication.
                  We break the 2x2 matrices into 4 scalars.
                  We perform 7 scalar multiplications.
                  
                  So, for each of the 7 block multiplications (s), we have a 2x2 matrix X_s and Y_s.
                  X_s is formed by linear combinations of the 4 blocks of A.
                  Each block of A is a 2x2 matrix.
                  So X_s is a 2x2 matrix of 2x2 matrices? No.
                  X_s is a 2x2 matrix. Its entries are 2x2 matrices (blocks).
                  Wait.
                  A00 is a 2x2 block.
                  X = A00 + A11. X is a 2x2 matrix.
                  X[0,0] = A[0,0] + A[2,2].
                  X[0,1] = A[0,1] + A[2,3].
                  etc.
                  
                  So X is a 2x2 matrix.
                  We need to decompose X * Y using Strassen 2x2.
                  The Strassen 2x2 algorithm takes a 2x2 matrix X and 2x2 matrix Y.
                  It forms 7 linear combinations of the *entries* of X and Y.
                  Entry X[0,0] is a scalar? No, in the context of the inner Strassen, X[0,0] is a scalar.
                  But here X[0,0] is a 2x2 matrix (block).
                  So we are effectively doing a 2x2x2 multiplication where the "scalars" are actually 2x2 matrices?
                  No, that would be recursive.
                  
                  Let's re-evaluate.
                  The rank of 4x4x4 is 49.
                  This implies we can decompose it into 49 rank-1 terms.
                  Each term is $u \otimes v \otimes w$ where $u, v, w$ are vectors of length 16, 16, 16.
                  
                  The structure is:
                  7 block multiplications.
                  Each block multiplication is a 2x2x2 multiplication.
                  Each 2x2x2 multiplication has rank 7.
                  So 7 * 7 = 49.
                  
                  In the inner 2x2x2 multiplication, the "scalars" are the elements of the 2x2 blocks.
                  So for a specific block multiplication s, we have matrices $X_s$ and $Y_s$ which are 2x2.
                  The entries of $X_s$ are linear combinations of the entries of A.
                  The entries of $Y_s$ are linear combinations of the entries of B.
                  Then we apply the 7 scalar multiplications of Strassen 2x2 to the entries of $X_s$ and $Y_s$.
                  
                  So, for each s (0..6) and t (0..6), we have a scalar multiplication.
                  $M_{s,t} = (\sum_{i,j} u_{s,t,i,j} A_{i,j}) \times (\sum_{j,k} v_{s,t,j,k} B_{j,k})$.
                  And this contributes to the output C.
                  The output of the inner Strassen is a 2x2 matrix $Z_s$.
                  $Z_s$ is a block of C.
                  The entries of $Z_s$ are linear combinations of the 7 products $M_{s,t}$.
                  So we can determine the W coefficients.
                  """
        pass

    # This is getting complex to implement generically. 
    # Let's hardcode the coefficients for the 49 terms.
    
    # We will generate the U, V, W for the 49 Strassen terms.
    
    # Define the 7 block combinations.
    # X_s and Y_s are 2x2 matrices.
    # X_s[r][c] is a linear combination of A blocks.
    # A block A_rc is a 2x2 matrix.
    # So X_s[r][c] is a 2x2 matrix of coefficients for A.
    
    # Let's define X_s_coeffs[s][r][c][u][v] = coeff for A[2r+u, 2c+v].
    # Similarly for Y_s_coeffs[s][c][d][v][w] = coeff for B[2c+v, 2d+w].
    
    # And Z_s_coeffs[s][r][d][u][w] = coeff for C[2r+u, 2d+w] from block M_s.
    
    # The 7 block multiplications:
    # s=0: M1 = (A00 + A11)(B00 + B11)
    # s=1: M2 = (A10 + A11)(B00)
    # s=2: M3 = (A00)(B10 - B11)
    # s=3: M4 = (A11)(B01 - B00)
    # s=4: M5 = (A00 + A10)(B11)
    # s=5: M6 = (A11 - A00)(B00 + B10)
    # s=6: M7 = (A01 - A11)(B11 + B01)
    
    # For each s, we define X_s and Y_s as 2x2 matrices of 2x2 matrices.
    # Actually, X_s is a 2x2 matrix where each entry is a 2x2 matrix.
    # Let's represent X_s as a 2x2x2x2 array: X_s[r][c][u][v].
    
    X_coeffs = [None] * 7
    Y_coeffs = [None] * 7
    Z_coeffs = [None] * 7
    
    # Initialize with zeros
    for s in range(7):
        X_coeffs[s] = [[[ [0]*2 for _ in range(2)] for _ in range(2)] for _ in range(2)]
        Y_coeffs[s] = [[[ [0]*2 for _ in range(2)] for _ in range(2)] for _ in range(2)]
        Z_coeffs[s] = [[[ [0]*2 for _ in range(2)] for _ in range(2)] for _ in range(2)]
        
    # s=0: M1 = (A00 + A11)(B00 + B11)
    # X0 = A00 + A11
    for u in range(2):
        for v in range(2):
            X_coeffs[0][0][0][u][v] = 1 # A00
            X_coeffs[0][1][1][u][v] = 1 # A11
            # Wait, X is 2x2. X[0,0] is A00? No.
            # X is the matrix A00 + A11.
            # X[0,0] = A[0,0] + A[2,2]
            # X[0,1] = A[0,1] + A[2,3]
            # X[1,0] = A[1,0] + A[3,2]
            # X[1,1] = A[1,1] + A[3,3]
            # So X[0,0] corresponds to block A00? No.
            # A00 is the top-left 2x2 block.
            # A11 is the bottom-right 2x2 block.
            # So X = A00 + A11.
            # X[0,0] = A[0,0] + A[2,2]
            # X[0,1] = A[0,1] + A[2,3]
            # X[1,0] = A[1,0] + A[3,2]
            # X[1,1] = A[1,1] + A[3,3]
            
            # So for X_coeffs[0], we need to set the coefficients for A.
            # X[0,0] gets 1 from A[0,0] and 1 from A[2,2].
            # But X_coeffs is indexed by [r][c][u][v] where r,c are block indices?
            # No, X is a 2x2 matrix. Its entries are scalars in the inner Strassen.
            # But the entries are sums of A elements.
            
            # Let's restructure.
            # For a given s, X_s is a 2x2 matrix.
            # Entry X_s[r][c] (where r,c in 0..1) is a linear combination of A elements.
            # Specifically, X_s[r][c] = sum_{i,j} coeff_{s,r,c,i,j} A[i,j].
            # We can store this as a 2x2x4x4 array? Or just a dict.
            
    # This is getting messy. Let's use a simpler representation.
    # For each s, define X_s as a 2x2 matrix of vectors of length 16 (flattened A).
    # Same for Y_s.
    # And Z_s as a 2x2 matrix of vectors of length 16 (flattened C).
    
    # Let's do that.
    
    X_vecs = [None] * 7
    Y_vecs = [None] * 7
    Z_vecs = [None] * 7
    
    for s in range(7):
        X_vecs[s] = [[0]*16 for _ in range(4)] # 2x2 matrix of vectors
        Y_vecs[s] = [[0]*16 for _ in range(4)]
        Z_vecs[s] = [[0]*16 for _ in range(4)]
        
    # Helper to set vector
    def set_vec(vec, i, j, coeff):
        vec[i*4 + j] += coeff
        
    # s=0: M1 = (A00 + A11)(B00 + B11)
    # X = A00 + A11
    for u in range(2):
        for v in range(2):
            set_vec(X_vecs[0][0], u, v, 1) # A[u,v]
            set_vec(X_vecs[0][0], 2+u, 2+v, 1) # A[2+u, 2+v]
            set_vec(X_vecs[0][1], u, 1+v, 1) # A[u, 1+v]
            set_vec(X_vecs[0][1], 2+u, 3+v, 1) # A[2+u, 3+v]
            set_vec(X_vecs[0][2], 1+u, v, 1) # A[1+u, v]
            set_vec(X_vecs[0][2], 3+u, 2+v, 1) # A[3+u, 2+v]
            set_vec(X_vecs[0][3], 1+u, 1+v, 1) # A[1+u, 1+v]
            set_vec(X_vecs[0][3], 3+u, 3+v, 1) # A[3+u, 3+v]
            
    # Wait, X is 2x2.
    # X[0,0] = A[0,0] + A[2,2]
    # X[0,1] = A[0,1] + A[2,3]
    # X[1,0] = A[1,0] + A[3,2]
    # X[1,1] = A[1,1] + A[3,3]
    
    # My loop above sets X_vecs[0][0] (which is X[0,0]) to A[0,0]+A[2,2] etc?
    # No, I mixed up indices.
    
    # Let's be precise.
    # X[0,0] is the top-left element of the 2x2 matrix X.
    # X[0,0] = A[0,0] + A[2,2].
    # So X_vecs[0][0] should have 1 at index 0 and 1 at index 10 (2*4+2).
    
    def add_to_vec(vec, i, j, coeff):
        vec[i*4 + j] += coeff
        
    # Reset
    for s in range(7):
        X_vecs[s] = [[0]*16 for _ in range(4)]
        Y_vecs[s] = [[0]*16 for _ in range(4)]
        Z_vecs[s] = [[0]*16 for _ in range(4)]
        
    # s=0
    # X[0,0] = A[0,0] + A[2,2]
    add_to_vec(X_vecs[0][0], 0, 0, 1); add_to_vec(X_vecs[0][0], 2, 2, 1)
    # X[0,1] = A[0,1] + A[2,3]
    add_to_vec(X_vecs[0][1], 0, 1, 1); add_to_vec(X_vecs[0][1], 2, 3, 1)
    # X[1,0] = A[1,0] + A[3,2]
    add_to_vec(X_vecs[0][2], 1, 0, 1); add_to_vec(X_vecs[0][2], 3, 2, 1)
    # X[1,1] = A[1,1] + A[3,3]
    add_to_vec(X_vecs[0][3], 1, 1, 1); add_to_vec(X_vecs[0][3], 3, 3, 1)
    
    # Y[0,0] = B[0,0] + B[2,2]
    add_to_vec(Y_vecs[0][0], 0, 0, 1); add_to_vec(Y_vecs[0][0], 2, 2, 1)
    # Y[0,1] = B[0,1] + B[2,3]
    add_to_vec(Y_vecs[0][1], 0, 1, 1); add_to_vec(Y_vecs[0][1], 2, 3, 1)
    # Y[1,0] = B[1,0] + B[3,2]
    add_to_vec(Y_vecs[0][2], 1, 0, 1); add_to_vec(Y_vecs[0][2], 3, 2, 1)
    # Y[1,1] = B[1,1] + B[3,3]
    add_to_vec(Y_vecs[0][3], 1, 1, 1); add_to_vec(Y_vecs[0][3], 3, 3, 1)
    
    # Z for s=0 (M1) contributes to C00 and C11.
    # C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    # So Z_s is the matrix M_s.
    # Z[0,0] = M1[0,0]
    # Z[0,1] = M1[0,1]
    # Z[1,0] = M1[1,0]
    # Z[1,1] = M1[1,1]
    # M1 is a 2x2 matrix.
    # M1[0,0] is the top-left element of the product of the 2x2 blocks.
    # But we are decomposing M1 = X Y using Strassen 2x2.
    # So the "result" of the inner Strassen is the 2x2 matrix M1.
    # The entries of M1 are linear combinations of the 7 scalar products.
    # We will handle the W matrix later.
    
    # For now, just define X and Y for all s.
    
    # s=1: M2 = (A10 + A11)(B00)
    # X[0,0] = A[2,0] + A[2,2] ? No.
    # A10 is block (1,0). A11 is block (1,1).
    # X = A10 + A11.
    # X[0,0] = A[2,0] + A[2,2]
    # X[0,1] = A[2,1] + A[2,3]
    # X[1,0] = A[3,0] + A[3,2]
    # X[1,1] = A[3,1] + A[3,3]
    add_to_vec(X_vecs[1][0], 2, 0, 1); add_to_vec(X_vecs[1][0], 2, 2, 1)
    add_to_vec(X_vecs[1][1], 2, 1, 1); add_to_vec(X_vecs[1][1], 2, 3, 1)
    add_to_vec(X_vecs[1][2], 3, 0, 1); add_to_vec(X_vecs[1][2], 3, 2, 1)
    add_to_vec(X_vecs[1][3], 3, 1, 1); add_to_vec(X_vecs[1][3], 3, 3, 1)
    
    # Y = B00.
    # Y[0,0] = B[0,0]
    # Y[0,1] = B[0,1]
    # Y[1,0] = B[1,0]
    # Y[1,1] = B[1,1]
    add_to_vec(Y_vecs[1][0], 0, 0, 1)
    add_to_vec(Y_vecs[1][1], 0, 1, 1)
    add_to_vec(Y_vecs[1][2], 1, 0, 1)
    add_to_vec(Y_vecs[1][3], 1, 1, 1)
    
    # s=2: M3 = (A00)(B10 - B11)
    # X = A00.
    add_to_vec(X_vecs[2][0], 0, 0, 1)
    add_to_vec(X_vecs[2][1], 0, 1, 1)
    add_to_vec(X_vecs[2][2], 1, 0, 1)
    add_to_vec(X_vecs[2][3], 1, 1, 1)
    
    # Y = B10 - B11.
    # B10 is block (1,0). B11 is block (1,1).
    # Y[0,0] = B[2,0] - B[2,2]
    # Y[0,1] = B[2,1] - B[2,3]
    # Y[1,0] = B[3,0] - B[3,2]
    # Y[1,1] = B[3,1] - B[3,3]
    add_to_vec(Y_vecs[2][0], 2, 0, 1); add_to_vec(Y_vecs[2][0], 2, 2, -1)
    add_to_vec(Y_vecs[2][1], 2, 1, 1); add_to_vec(Y_vecs[2][1], 2, 3, -1)
    add_to_vec(Y_vecs[2][2], 3, 0, 1); add_to_vec(Y_vecs[2][2], 3, 2, -1)
    add_to_vec(Y_vecs[2][3], 3, 1, 1); add_to_vec(Y_vecs[2][3], 3, 3, -1)
    
    # s=3: M4 = (A11)(B01 - B00)
    # X = A11.
    add_to_vec(X_vecs[3][0], 2, 2, 1)
    add_to_vec(X_vecs[3][1], 2, 3, 1)
    add_to_vec(X_vecs[3][2], 3, 2, 1)
    add_to_vec(X_vecs[3][3], 3, 3, 1)
    
    # Y = B01 - B00.
    # B01 is block (0,1). B00 is block (0,0).
    # Y[0,0] = B[0,2] - B[0,0]
    # Y[0,1] = B[0,3] - B[0,1]
    # Y[1,0] = B[1,2] - B[1,0]
    # Y[1,1] = B[1,3] - B[1,1]
    add_to_vec(Y_vecs[3][0], 0, 2, 1); add_to_vec(Y_vecs[3][0], 0, 0, -1)
    add_to_vec(Y_vecs[3][1], 0, 3, 1); add_to_vec(Y_vecs[3][1], 0, 1, -1)
    add_to_vec(Y_vecs[3][2], 1, 2, 1); add_to_vec(Y_vecs[3][2], 1, 0, -1)
    add_to_vec(Y_vecs[3][3], 1, 3, 1); add_to_vec(Y_vecs[3][3], 1, 1, -1)
    
    # s=4: M5 = (A00 + A10)(B11)
    # X = A00 + A10.
    # X[0,0] = A[0,0] + A[2,0]
    # X[0,1] = A[0,1] + A[2,1]
    # X[1,0] = A[1,0] + A[3,0]
    # X[1,1] = A[1,1] + A[3,1]
    add_to_vec(X_vecs[4][0], 0, 0, 1); add_to_vec(X_vecs[4][0], 2, 0, 1)
    add_to_vec(X_vecs[4][1], 0, 1, 1); add_to_vec(X_vecs[4][1], 2, 1, 1)
    add_to_vec(X_vecs[4][2], 1, 0, 1); add_to_vec(X_vecs[4][2], 3, 0, 1)
    add_to_vec(X_vecs[4][3], 1, 1, 1); add_to_vec(X_vecs[4][3], 3, 1, 1)
    
    # Y = B11.
    add_to_vec(Y_vecs[4][0], 2, 2, 1)
    add_to_vec(Y_vecs[4][1], 2, 3, 1)
    add_to_vec(Y_vecs[4][2], 3, 2, 1)
    add_to_vec(Y_vecs[4][3], 3, 3, 1)
    
    # s=5: M6 = (A11 - A00)(B00 + B10)
    # X = A11 - A00.
    # X[0,0] = A[2,2] - A[0,0]
    # X[0,1] = A[2,3] - A[0,1]
    # X[1,0] = A[3,2] - A[1,0]
    # X[1,1] = A[3,3] - A[1,1]
    add_to_vec(X_vecs[5][0], 2, 2, 1); add_to_vec(X_vecs[5][0], 0, 0, -1)
    add_to_vec(X_vecs[5][1], 2, 3, 1); add_to_vec(X_vecs[5][1], 0, 1, -1)
    add_to_vec(X_vecs[5][2], 3, 2, 1); add_to_vec(X_vecs[5][2], 1, 0, -1)
    add_to_vec(X_vecs[5][3], 3, 3, 1); add_to_vec(X_vecs[5][3], 1, 1, -1)
    
    # Y = B00 + B10.
    # Y[0,0] = B[0,0] + B[2,0]
    # Y[0,1] = B[0,1] + B[2,1]
    # Y[1,0] = B[1,0] + B[3,0]
    # Y[1,1] = B[1,1] + B[3,1]
    add_to_vec(Y_vecs[5][0], 0, 0, 1); add_to_vec(Y_vecs[5][0], 2, 0, 1)
    add_to_vec(Y_vecs[5][1], 0, 1, 1); add_to_vec(Y_vecs[5][1], 2, 1, 1)
    add_to_vec(Y_vecs[5][2], 1, 0, 1); add_to_vec(Y_vecs[5][2], 3, 0, 1)
    add_to_vec(Y_vecs[5][3], 1, 1, 1); add_to_vec(Y_vecs[5][3], 3, 1, 1)
    
    # s=6: M7 = (A01 - A11)(B11 + B01)
    # X = A01 - A11.
    # X[0,0] = A[0,2] - A[2,2]
    # X[0,1] = A[0,3] - A[2,3]
    # X[1,0] = A[1,2] - A[3,2]
    # X[1,1] = A[1,3] - A[3,3]
    add_to_vec(X_vecs[6][0], 0, 2, 1); add_to_vec(X_vecs[6][0], 2, 2, -1)
    add_to_vec(X_vecs[6][1], 0, 3, 1); add_to_vec(X_vecs[6][1], 2, 3, -1)
    add_to_vec(X_vecs[6][2], 1, 2, 1); add_to_vec(X_vecs[6][2], 3, 2, -1)
    add_to_vec(X_vecs[6][3], 1, 3, 1); add_to_vec(X_vecs[6][3], 3, 3, -1)
    
    # Y = B11 + B01.
    # Y[0,0] = B[2,2] + B[0,2]
    # Y[0,1] = B[2,3] + B[0,3]
    # Y[1,0] = B[3,2] + B[1,2]
    # Y[1,1] = B[3,3] + B[1,3]
    add_to_vec(Y_vecs[6][0], 2, 2, 1); add_to_vec(Y_vecs[6][0], 0, 2, 1)
    add_to_vec(Y_vecs[6][1], 2, 3, 1); add_to_vec(Y_vecs[6][1], 0, 3, 1)
    add_to_vec(Y_vecs[6][2], 3, 2, 1); add_to_vec(Y_vecs[6][2], 1, 2, 1)
    add_to_vec(Y_vecs[6][3], 3, 3, 1); add_to_vec(Y_vecs[6][3], 1, 3, 1)
    
    # Now, for each s, we apply Strassen 2x2 to X_vecs[s] and Y_vecs[s].
    # This generates 7 scalar multiplications for each s.
    # Total 49 multiplications.
    
    # The 7 Strassen 2x2 products:
    # P1 = (X00 + X11)(Y00 + Y11)
    # P2 = (X10 + X11)(Y00)
    # P3 = (X00)(Y10 - Y11)
    # P4 = (X11)(Y01 - Y00)
    # P5 = (X00 + X10)(Y11)
    # P6 = (X11 - X00)(Y00 + Y10)
    # P7 = (X01 - X11)(Y11 + Y01)
    
    # Result Z:
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # We will fill U, V, W for these 49 terms.
    
    for s in range(7):
        X = X_vecs[s]
        Y = Y_vecs[s]
        
        # Define the 7 linear combinations for X and Y.
        # X00, X01, X10, X11 are vectors of length 16.
        # We need to form the sums.
        
        def vec_add(a, b):
            return [x + y for x, y in zip(a, b)]
        
        def vec_sub(a, b):
            return [x - y for x, y in zip(a, b)]
        
        def vec_mul_scalar(a, c):
            return [x * c for x in a]
            
        # P1
        u1 = vec_add(X[0], X[3])
        v1 = vec_add(Y[0], Y[3])
        # P2
        u2 = vec_add(X[2], X[3])
        v2 = Y[0]
        # P3
        u3 = X[0]
        v3 = vec_sub(Y[2], Y[3])
        # P4
        u4 = X[3]
        v4 = vec_sub(Y[1], Y[0])
        # P5
        u5 = vec_add(X[0], X[2])
        v5 = Y[3]
        # P6
        u6 = vec_sub(X[3], X[0])
        v6 = vec_add(Y[0], Y[2])
        # P7
        u7 = vec_sub(X[1], X[3])
        v7 = vec_add(Y[3], Y[1])
        
        products = [
            (u1, v1),
            (u2, v2),
            (u3, v3),
            (u4, v4),
            (u5, v5),
            (u6, v6),
            (u7, v7)
        ]
        
        # Z coefficients
        # Z00 = P1 + P4 - P5 + P7
        # Z01 = P3 + P5
        # Z10 = P2 + P4
        # Z11 = P1 - P2 + P3 + P6
        
        # We need to map Z[0,0] etc to the output C indices.
        # Z is a 2x2 matrix.
        # Z[0,0] corresponds to C block (0,0).
        # Z[0,1] corresponds to C block (0,1).
        # Z[1,0] corresponds to C block (1,0).
        # Z[1,1] corresponds to C block (1,1).
        
        # C block (r,d) is a 2x2 matrix.
        # C[2r+u, 2d+w].
        
        # For each product t (0..6), we add a term to U, V, W.
        for t in range(7):
            u_t, v_t = products[t]
            
            # Add to U and V
            for idx in range(16):
                U[idx][col] = u_t[idx]
                V[idx][col] = v_t[idx]
                
            # Determine W coefficients.
            # We need to know which Z entry this product contributes to.
            # The contribution of P_t to Z is:
            # If t=0 (P1): Z00 += 1, Z11 += 1
            # If t=1 (P2): Z10 += 1, Z11 += -1
            # If t=2 (P3): Z01 += 1, Z11 += 1
            # If t=3 (P4): Z00 += 1, Z10 += 1
            # If t=4 (P5): Z00 += -1, Z01 += 1
            # If t=5 (P6): Z11 += 1
            # If t=6 (P7): Z00 += 1
            
            # But wait, the Z here is the block M_s.
            # And M_s contributes to C with coefficients from the outer Strassen.
            # Outer Strassen:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So we need to combine the inner Z coefficients with the outer C coefficients.
            
            # Let's define the inner Z coefficients for each t.
            # z_inner[t][r][d]
            
            z_inner = [[0]*2 for _ in range(2)]
            if t == 0: # P1
                z_inner[0][0] += 1
                z_inner[1][1] += 1
            elif t == 1: # P2
                z_inner[1][0] += 1
                z_inner[1][1] -= 1
            elif t == 2: # P3
                z_inner[0][1] += 1
                z_inner[1][1] += 1
            elif t == 3: # P4
                z_inner[0][0] += 1
                z_inner[1][0] += 1
            elif t == 4: # P5
                z_inner[0][0] -= 1
                z_inner[0][1] += 1
            elif t == 5: # P6
                z_inner[1][1] += 1
            elif t == 6: # P7
                z_inner[0][0] += 1
            
            # Now apply outer coefficients.
            # C00 (block 0,0) gets M1(1), M4(1), M5(-1), M7(1)
            # C01 (block 0,1) gets M3(1), M5(1)
            # C10 (block 1,0) gets M2(1), M4(1)
            # C11 (block 1,1) gets M1(1), M2(-1), M3(1), M6(1)
            
            # Outer coeff for M_s to C_rd:
            outer_coeff = [[0]*2 for _ in range(2)]
            if s == 0: # M1
                outer_coeff[0][0] = 1
                outer_coeff[1][1] = 1
            elif s == 1: # M2
                outer_coeff[1][0] = 1
                outer_coeff[1][1] = -1
            elif s == 2: # M3
                outer_coeff[0][1] = 1
                outer_coeff[1][1] = 1
            elif s == 3: # M4
                outer_coeff[0][0] = 1
                outer_coeff[1][0] = 1
            elif s == 4: # M5
                outer_coeff[0][0] = -1
                outer_coeff[0][1] = 1
            elif s == 5: # M6
                outer_coeff[1][1] = 1
            elif s == 6: # M7
                outer_coeff[0][0] = 1
            
            # Total W coefficient for C[2r+u, 2d+w] is z_inner[r][d] * outer_coeff[r][d]
            for r in range(2):
                for d in range(2):
                    coeff = z_inner[r][d] * outer_coeff[r][d]
                    if coeff != 0:
                        for u in range(2):
                            for w in range(2):
                                i = 2*r + u
                                k = 2*d + w
                                W[k*4 + i][col] += coeff
            
            col += 1
            
    # Add schoolbook for 5th column (k=4)
    for i in range(4):
        for j in range(4):
            U[i*4 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
