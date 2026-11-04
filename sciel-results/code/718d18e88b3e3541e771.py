import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Indices for A (4x4): i in 0..3, j in 0..3. Index = i*4 + j.
    # Indices for B (4x4): j in 0..3, k in 0..3. Index = j*4 + k.
    # Indices for C (4x4): i in 0..3, k in 0..3. Index = k*4 + i (Trace convention W[k*n+i]).
    # Wait, W[k*n+i] means row index is k*4 + i.
    # C[i, k] corresponds to W[k*4 + i].
    
    # Outer Strassen definitions
    # A_blocks: dict from block_name to list of (coeff, i, j)
    # B_blocks: dict from block_name to list of (coeff, j, k)
    
    # We define the 7 outer terms
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Inner Strassen for X*Y = Z
    # S1: (X00+X11)(Y00+Y11)
    # S2: (X10+X11)Y00
    # S3: X00(Y10-Y11)
    # S4: X11(Y01-Y00)
    # S5: (X00+X01)Y11
    # S6: (X10-X00)(Y00+Y01)
    # S7: (X01-X11)(Y10+Y11)
    
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map each S_kl to C_ij.
    # Let's create a dictionary coeff_C[i][k] for each term.
    
    # First, determine the linear forms for A and B for each of the 49 terms.
    # Then, determine the coefficient in C.
    
    # Helper to add to U, V, W
    def add_term(U_idx, V_idx, W_idx, coeff, col):
        U[U_idx, col] += coeff
        V[V_idx, col] += coeff
        W[W_idx, col] += coeff

    # This approach is getting complicated to trace manually in code.
    # Let's generate the terms programmatically.
    
    # Define A block linear forms for each M
    # A_M[k] is a 4x4 matrix of coefficients for A
    # B_M[k] is a 4x4 matrix of coefficients for B
    
    A_M = [np.zeros((4,4)) for _ in range(7)]
    B_M = [np.zeros((4,4)) for _ in range(7)]
    
    # M1: (A00+A11)(B00+B11)
    A_M[0][0,0] = 1; A_M[0][0,1] = 1; A_M[0][1,0] = 1; A_M[0][1,1] = 1
    A_M[0][2,2] = 1; A_M[0][2,3] = 1; A_M[0][3,2] = 1; A_M[0][3,3] = 1
    B_M[0][0,0] = 1; B_M[0][0,1] = 1; B_M[0][1,0] = 1; B_M[0][1,1] = 1
    B_M[0][2,2] = 1; B_M[0][2,3] = 1; B_M[0][3,2] = 1; B_M[0][3,3] = 1
    
    # M2: (A10+A11)B00
    A_M[1][2,0] = 1; A_M[1][2,1] = 1; A_M[1][3,0] = 1; A_M[1][3,1] = 1
    A_M[1][2,2] = 1; A_M[1][2,3] = 1; A_M[1][3,2] = 1; A_M[1][3,3] = 1
    B_M[1][0,0] = 1; B_M[1][0,1] = 1; B_M[1][1,0] = 1; B_M[1][1,1] = 1
    
    # M3: A00(B10-B11)
    A_M[2][0,0] = 1; A_M[2][0,1] = 1; A_M[2][1,0] = 1; A_M[2][1,1] = 1
    B_M[2][2,0] = 1; B_M[2][2,1] = 1; B_M[2][3,0] = 1; B_M[2][3,1] = 1
    B_M[2][2,2] = -1; B_M[2][2,3] = -1; B_M[2][3,2] = -1; B_M[2][3,3] = -1
    
    # M4: A11(B01-B00)
    A_M[3][2,2] = 1; A_M[3][2,3] = 1; A_M[3][3,2] = 1; A_M[3][3,3] = 1
    B_M[3][0,1] = 1; B_M[3][1,1] = 1; B_M[3][0,0] = -1; B_M[3][1,0] = -1
    
    # M5: (A00+A01)B11
    A_M[4][0,0] = 1; A_M[4][0,1] = 1; A_M[4][1,0] = 1; A_M[4][1,1] = 1
    A_M[4][0,2] = 1; A_M[4][0,3] = 1; A_M[4][1,2] = 1; A_M[4][1,3] = 1
    B_M[4][2,2] = 1; B_M[4][2,3] = 1; B_M[4][3,2] = 1; B_M[4][3,3] = 1
    
    # M6: (A10-A00)(B00+B01)
    A_M[5][2,0] = 1; A_M[5][2,1] = 1; A_M[5][3,0] = 1; A_M[5][3,1] = 1
    A_M[5][0,0] = -1; A_M[5][0,1] = -1; A_M[5][1,0] = -1; A_M[5][1,1] = -1
    B_M[5][0,0] = 1; B_M[5][0,1] = 1; B_M[5][1,0] = 1; B_M[5][1,1] = 1
    
    # M7: (A01-A11)(B10+B11)
    A_M[6][0,2] = 1; A_M[6][0,3] = 1; A_M[6][1,2] = 1; A_M[6][1,3] = 1
    A_M[6][2,2] = -1; A_M[6][2,3] = -1; A_M[6][3,2] = -1; A_M[6][3,3] = -1
    B_M[6][2,0] = 1; B_M[6][2,1] = 1; B_M[6][3,0] = 1; B_M[6][3,1] = 1
    
    # Now, for each M_k, we apply inner Strassen.
    # Inner Strassen uses X and Y which are 2x2 blocks of the "effective" matrices.
    # The "effective" A matrix for M_k is A_M[k].
    # The "effective" B matrix for M_k is B_M[k].
    # But Inner Strassen operates on 2x2 matrices.
    # The blocks of A_M[k] are 2x2 submatrices.
    # A_M[k] is 4x4. Its blocks are A_M[k][0:2, 0:2], etc.
    
    col = 0
    for k in range(7):
        # Extract 2x2 blocks of A_M[k] and B_M[k]
        # A_M[k] is the coefficient matrix for the linear combination.
        # Let's define the 4 blocks of A_M[k]:
        # AB00 = A_M[k][0:2, 0:2]
        # AB01 = A_M[k][0:2, 2:4]
        # AB10 = A_M[k][2:4, 0:2]
        # AB11 = A_M[k][2:4, 2:4]
        
        AB00 = A_M[k][0:2, 0:2]
        AB01 = A_M[k][0:2, 2:4]
        AB10 = A_M[k][2:4, 0:2]
        AB11 = A_M[k][2:4, 2:4]
        
        BB00 = B_M[k][0:2, 0:2]
        BB01 = B_M[k][0:2, 2:4]
        BB10 = B_M[k][2:4, 0:2]
        BB11 = B_M[k][2:4, 2:4]
        
        # Inner Strassen terms
        # S1: (X00+X11)(Y00+Y11)
        # X00 is the top-left 2x2 block of the "X" matrix.
        # But X is a 2x2 matrix of scalars? No, X is a 2x2 matrix of 2x2 blocks?
        # No, in the recursive step, we treat the 4x4 matrix as 2x2 blocks.
        # So X is a 2x2 matrix where each entry is a 2x2 block of A.
        # So X00 is the top-left 2x2 block of A_M[k].
        
        # Inner Strassen on 2x2 matrix of blocks.
        # S1 = (X00 + X11) * (Y00 + Y11)
        # This is a product of two 2x2 block matrices.
        # But we want scalar multiplications.
        # So we expand (X00 + X11) and (Y00 + Y11) into linear forms of scalars.
        # And then the product of two linear forms is not a single scalar multiplication.
        # Wait. The rank-1 term is a scalar multiplication $u \cdot v$.
        # $u$ is a linear form in A, $v$ is a linear form in B.
        # So $S_1$ is NOT a single rank-1 term.
        # $S_1$ is the product of two linear forms?
        # No, Strassen replaces $AB$ with 7 products of linear forms.
        # So $S_1$ IS a product of two linear forms.
        # $L_A = \text{linear form from A}$
        # $L_B = \text{linear form from B}$
        # $S_1 = L_A L_B$.
        
        # So for each inner term, we need to find $L_A$ and $L_B$.
        
        # S1: L_A = X00 + X11, L_B = Y00 + Y11
        # X00 is the sum of entries in A_M[k][0:2, 0:2] weighted by A?
        # No, X00 is a 2x2 block.
        # The linear form $L_A$ for S1 is the sum of all entries in X00 and X11?
        # No, $S_1 = (X_{00} + X_{11})(Y_{00} + Y_{11})$.
        # $X_{00}$ is a scalar? No, $X$ is a 2x2 matrix of scalars?
        # In the inner Strassen, we are multiplying two 2x2 matrices of scalars.
        # The "blocks" of the 4x4 matrix are 2x2 matrices of scalars.
        # So $X_{00}$ is a 2x2 matrix of scalars.
        # So $(X_{00} + X_{11})$ is a 2x2 matrix of scalars.
        # And $(Y_{00} + Y_{11})$ is a 2x2 matrix of scalars.
        # Their product is a 2x2 matrix of scalars.
        # This product is computed using 7 scalar multiplications (Inner Strassen).
        
        # So, for each of the 7 inner terms, we have a scalar multiplication.
        # We need to find the linear form in A and B for each scalar multiplication.
        
        # Let's define the 7 inner linear forms for A and B.
        # For a generic 2x2 matrix X (entries x00, x01, x10, x11) and Y (y00, y01, y10, y11):
        # S1: (x00+x11)(y00+y11)
        # S2: (x10+x11)y00
        # S3: x00(y10-y11)
        # S4: x11(y01-y00)
        # S5: (x00+x01)y11
        # S6: (x10-x00)(y00+y01)
        # S7: (x01-x11)(y10+y11)
        
        # Here, x00, x01, etc. are the entries of the 2x2 block matrix X.
        # But X is formed from the 4x4 matrix A_M[k].
        # X is a 2x2 matrix of 2x2 blocks.
        # So x00 is the top-left 2x2 block of A_M[k].
        # So x00 is a 2x2 matrix of coefficients.
        # The linear form for S1's A-part is the sum of all entries in x00 and x11?
        # No, $S_1 = (x_{00} + x_{11})(y_{00} + y_{11})$.
        # This is a product of two 2x2 matrices.
        # This is NOT a scalar product.
        
        # I am confusing the levels.
        # Level 1: 4x4 matrix multiplication. Decomposed into 7 products of 2x2 block matrices.
        # Level 2: 2x2 block matrix multiplication. Decomposed into 7 scalar products.
        
        # So, for each of the 7 Level-1 products, we perform Level-2 Strassen.
        # The Level-2 Strassen produces 7 scalar products.
        # Total 49 scalar products.
        
        # For each scalar product, we need:
        # 1. The linear form in A (coefficients for $a_{ij}$).
        # 2. The linear form in B (coefficients for $b_{jk}$).
        # 3. The contribution to each $c_{ik}$.
        
        # Let's trace one scalar product.
        # Take M1 = (A00+A11)(B00+B11).
        # X = A00+A11, Y = B00+B11.
        # X is a 2x2 matrix. X00 = A00_00 + A11_00 = a00 + a22.
        # X11 = A00_11 + A11_11 = a11 + a33.
        # S1 = (X00+X11)(Y00+Y11).
        # X00+X11 = a00+a22+a11+a33.
        # Y00+Y11 = b00+b22+b11+b33.
        # S1 = (a00+a11+a22+a33)(b00+b11+b22+b33).
        # This is ONE scalar multiplication?
        # No, Strassen S1 is $(x00+x11)(y00+y11)$.
        # $x00$ and $x11$ are scalars.
        # So yes, it is a product of two linear forms.
        # So $L_A = a00+a11+a22+a33$.
        # $L_B = b00+b11+b22+b33$.
        
        # So for each inner term, we can construct $L_A$ and $L_B$.
        
        # We need to map the inner term's result to C.
        # M1 contributes to C00, C11.
        # M1 = Z (2x2 matrix).
        # Z00 = S1+S4-S5+S7.
        # Z01 = S3+S5.
        # Z10 = S2+S4.
        # Z11 = S1-S2+S3+S6.
        
        # C00 = M1_00 + M4_00 - M5_00 + M7_00.
        # C11 = M1_11 - M2_11 + M3_11 + M6_11.
        
        # So we can sum up the coefficients for each S_kl in each C_ij.
        
        pass
