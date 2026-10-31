"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen rank for 4x4 is 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add coefficients
    def add_coeff(mat, row, col, val):
        if val != 0:
            mat[row][col] += val

    # Define Outer Strassen Coefficients
    # X_s and Y_s are 2x2 matrices.
    # X_s[u, v] = sum_{i,j} alpha A[i,j]
    # We represent alpha as a dict or 4x4 matrix.
    
    # Outer 0: X = A00 + A11, Y = B00 + B11
    # A00 is rows 0,1 cols 0,1. A11 is rows 2,3 cols 2,3.
    # X[0,0] = A[0,0]+A[2,2], X[0,1] = A[0,1]+A[2,3], etc.
    
    # We will iterate s in 0..6
    # For each s, define X_coeffs[u][v][i][j] and Y_coeffs[u][v][j][k]
    
    # X is 2x2, entries are linear combos of A (4x4)
    # Y is 2x2, entries are linear combos of B (4x4)
    
    # Outer Definitions:
    # s=0: X = A00+A11, Y = B00+B11
    # s=1: X = A10+A11, Y = B00
    # s=2: X = A00, Y = B01-B11
    # s=3: X = A11, Y = B10-B00
    # s=4: X = A00+A01, Y = B11
    # s=5: X = A10-A00, Y = B00+B01
    # s=6: X = A01-A11, Y = B10+B11
    
    # Blocks:
    # A00: i in 0,1; j in 0,1
    # A01: i in 0,1; j in 2,3
    # A10: i in 2,3; j in 0,1
    # A11: i in 2,3; j in 2,3
    
    # We build X_coeffs as a list of 4 matrices (one for each X entry 0,0; 0,1; 1,0; 1,1)
    # Each matrix is 4x4, containing coeffs for A[i,j].
    
    def get_block_coeffs(block_name):
        # block_name in {'00', '01', '10', '11'}
        # Returns a 4x4 matrix with 1s in the block, 0s elsewhere.
        mat = [[0]*4 for _ in range(4)]
        if block_name == '00':
            for i in [0,1]:
                for j in [0,1]:
                    mat[i][j] = 1
        elif block_name == '01':
            for i in [0,1]:
                for j in [2,3]:
                    mat[i][j] = 1
        elif block_name == '10':
            for i in [2,3]:
                for j in [0,1]:
                    mat[i][j] = 1
        elif block_name == '11':
            for i in [2,3]:
                for j in [2,3]:
                    mat[i][j] = 1
        return mat

    def add_mats(mat1, mat2, sign=1):
        res = [[0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                res[i][j] = mat1[i][j] + sign * mat2[i][j]
        return res

    def scale_mat(mat, sign):
        res = [[0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                res[i][j] = sign * mat[i][j]
        return res

    # Define X and Y block structures for each s
    # X_struct[s] is a list of 4 4x4 matrices (coeffs for X[0,0], X[0,1], X[1,0], X[1,1])
    # Y_struct[s] is a list of 4 4x4 matrices (coeffs for Y[0,0], Y[0,1], Y[1,0], Y[1,1])
    
    A00 = get_block_coeffs('00')
    A01 = get_block_coeffs('01')
    A10 = get_block_coeffs('10')
    A11 = get_block_coeffs('11')
    B00 = get_block_coeffs('00')
    B01 = get_block_coeffs('01')
    B10 = get_block_coeffs('10')
    B11 = get_block_coeffs('11')
    
    # Zero matrix
    Z = [[0]*4 for _ in range(4)]
    
    # s=0
    X0 = [add_mats(A00, A11), Z, Z, Z] # X[0,0]=A00+A11, others 0? 
    # Wait. X is a 2x2 matrix. 
    # M1 = (A00+A11)(B00+B11).
    # Here X is the matrix (A00+A11). This is a 2x2 matrix?
    # No. A00 is 2x2. A11 is 2x2.
    # The sum A00+A11 is a 2x2 matrix where element (0,0) is A00[0,0]+A11[0,0] etc.
    # So X[0,0] (the top-left element of the 2x2 matrix X) is the linear combo.
    # What are X[0,1], X[1,0], X[1,1]?
    # In the expression (A00+A11), this is a single 2x2 matrix.
    # So X = [[A00[0,0]+A11[0,0], A00[0,1]+A11[0,1]],
    #          [A00[1,0]+A11[1,0], A00[1,1]+A11[1,1]]]
    # So X[0,0] corresponds to the sum of the (0,0) elements of the blocks.
    # My `get_block_coeffs` returns a 4x4 matrix of coeffs for the whole A.
    # So `add_mats(A00, A11)` gives a 4x4 matrix with 1s at A[0,0], A[0,1], A[1,0], A[1,1] and A[2,2], A[2,3], A[3,2], A[3,3].
    # This represents the linear combo for the **entire** 2x2 matrix X?
    # No. X is a 2x2 matrix.
    # The entries of X are scalars.
    # X[0,0] = A[0,0] + A[2,2].
    # X[0,1] = A[0,1] + A[2,3].
    # X[1,0] = A[1,0] + A[3,2].
    # X[1,1] = A[1,1] + A[3,3].
    
    # So I need to map the 4x4 block coefficients to the 4 scalar entries of X.
    
    # Let's define X_coeffs[s][u][v] as a 4x4 matrix of coeffs for A.
    # For s=0:
    # X[0,0] coeffs: A[0,0]=1, A[2,2]=1. Others 0.
    # X[0,1] coeffs: A[0,1]=1, A[2,3]=1.
    # X[1,0] coeffs: A[1,0]=1, A[3,2]=1.
    # X[1,1] coeffs: A[1,1]=1, A[3,3]=1.
    
    # This is getting complicated to hardcode manually. I will write the logic to extract these.
    
    # Generic extraction:
    # For a block sum like A00+A11:
    # The resulting 2x2 matrix X has entries:
    # X[0,0] = A[0,0] + A[2,2]
    # X[0,1] = A[0,1] + A[2,3]
    # X[1,0] = A[1,0] + A[3,2]
    # X[1,1] = A[1,1] + A[3,3]
    
    # For A00 alone:
    # X[0,0]=A[0,0], X[0,1]=A[0,1], X[1,0]=A[1,0], X[1,1]=A[1,1]. Others 0.
    
    # For A00+A01:
    # X[0,0]=A[0,0], X[0,1]=A[0,1]+A[0,2], X[1,0]=A[1,0], X[1,1]=A[1,1]+A[1,2].
    
    # I will define the 4x4 coeff matrix for each scalar entry of X and Y.
    
    def make_X_Y(s):
        X_coeffs = [[[[0]*4 for _ in range(4)] for _ in range(2)] for _ in range(2)]
        Y_coeffs = [[[[0]*4 for _ in range(4)] for _ in range(2)] for _ in range(2)]
        
        # Helper to add block to X entry (u,v)
        def add_to_X(u, v, block, sign):
            # block is '00', '01', '10', '11'
            # u,v in {0,1}
            # Maps block local indices to global.
            # Block 00: global i=0,1; j=0,1. Local i=i, j=j.
            # Block 01: global i=0,1; j=2,3. Local i=i, j=j+2.
            # Block 10: global i=2,3; j=0,1. Local i=i+2, j=j.
            # Block 11: global i=2,3; j=2,3. Local i=i+2, j=j+2.
            
            # We add sign * A[global_i][global_j] to X_coeffs[u][v][global_i][global_j]
            # But X[u,v] is a scalar.
            # The block contributes to X[u,v] only if the local position matches?
            # No.
            # X is a 2x2 matrix.
            # If X = A00, then X[0,0]=A[0,0], X[0,1]=A[0,1], X[1,0]=A[1,0], X[1,1]=A[1,1].
            # So for X[0,0], coeff of A[0,0] is 1.
            # For X[0,1], coeff of A[0,1] is 1.
            
            # So for a given block and X entry (u,v):
            # The local indices in the block corresponding to X entry (u,v) are (u,v).
            # Global indices:
            if block == '00':
                gi, gj = u, v
            elif block == '01':
                gi, gj = u, v+2
            elif block == '10':
                gi, gj = u+2, v
            elif block == '11':
                gi, gj = u+2, v+2
            
            X_coeffs[u][v][gi][gj] += sign

        def add_to_Y(u, v, block, sign):
            # Y is 2x2. Block indices for B.
            # B00: i=0,1; j=0,1.
            # B01: i=0,1; j=2,3.
            # B10: i=2,3; j=0,1.
            # B11: i=2,3; j=2,3.
            # Y[u,v] gets contribution from B block at local (u,v).
            if block == '00':
                gi, gj = u, v
            elif block == '01':
                gi, gj = u, v+2
            elif block == '10':
                gi, gj = u+2, v
            elif block == '11':
                gi, gj = u+2, v+2
            
            Y_coeffs[u][v][gi][gj] += sign

        if s == 0:
            # X = A00 + A11
            for u in range(2):
                for v in range(2):
                    add_to_X(u, v, '00', 1)
                    add_to_X(u, v, '11', 1)
            # Y = B00 + B11
            for u in range(2):
                for v in range(2):
                    add_to_Y(u, v, '00', 1)
                    add_to_Y(u, v, '11', 1)
        elif s == 1:
            # X = A10 + A11
            for u in range(2):
                for v in range(2):
                    add_to_X(u, v, '10', 1)
                    add_to_X(u, v, '11', 1)
            # Y = B00
            for u in range(2):
                for v in range(2):
                    add_to_Y(u, v, '00', 1)
        elif s == 2:
            # X = A00
            for u in range(2):
                for v in range(2):
                    add_to_X(u, v, '00', 1)
            # Y = B01 - B11
            for u in range(2):
                for v in range(2):
                    add_to_Y(u, v, '01', 1)
                    add_to_Y(u, v, '11', -1)
        elif s == 3:
            # X = A11
            for u in range(2):
                for v in range(2):
                    add_to_X(u, v, '11', 1)
            # Y = B10 - B00
            for u in range(2):
                for v in range(2):
                    add_to_Y(u, v, '10', 1)
                    add_to_Y(u, v, '00', -1)
        elif s == 4:
            # X = A00 + A01
            for u in range(2):
                for v in range(2):
                    add_to_X(u, v, '00', 1)
                    add_to_X(u, v, '01', 1)
            # Y = B11
            for u in range(2):
                for v in range(2):
                    add_to_Y(u, v, '11', 1)
        elif s == 5:
            # X = A10 - A00
            for u in range(2):
                for v in range(2):
                    add_to_X(u, v, '10', 1)
                    add_to_X(u, v, '00', -1)
            # Y = B00 + B01
            for u in range(2):
                for v in range(2):
                    add_to_Y(u, v, '00', 1)
                    add_to_Y(u, v, '01', 1)
        elif s == 6:
            # X = A01 - A11
            for u in range(2):
                for v in range(2):
                    add_to_X(u, v, '01', 1)
                    add_to_X(u, v, '11', -1)
            # Y = B10 + B11
            for u in range(2):
                for v in range(2):
                    add_to_Y(u, v, '10', 1)
                    add_to_Y(u, v, '11', 1)
        
        return X_coeffs, Y_coeffs

    # Inner Strassen Coefficients for 2x2 matrices X, Y
    # S_t = (Linear Combo of X) * (Linear Combo of Y)
    # X entries: X00, X01, X10, X11
    # Y entries: Y00, Y01, Y10, Y11
    
    # t=0: (X00+X11)(Y00+Y11)
    # t=1: (X10+X11)Y00
    # t=2: X00(Y01-Y11)
    # t=3: X11(Y10-Y00)
    # t=4: (X00+X01)Y11
    # t=5: (X10-X00)(Y00+Y01)
    # t=6: (X01-X11)(Y10+Y11)
    
    inner_x_defs = [
        [[0,0,1,1], [0,0,0,0], [0,0,0,0], [0,0,0,0]], # t=0: X00+X11 -> coeffs for X00,X01,X10,X11?
        # No, X_coeffs is 2x2 of 4x4 matrices.
        # I need to combine the 4x4 matrices of X[0,0] and X[1,1].
        
    ]
    
    # Instead of pre-defining, I will compute the U and V vectors for each t.
    
    # Inner Strassen W contributions:
    # Z00 = S0 + S3 - S4 + S6
    # Z01 = S2 + S4
    # Z10 = S1 + S3
    # Z11 = S0 - S1 + S2 + S5
    
    # Z is a 2x2 matrix.
    # Z[0,0] corresponds to C block 00, local 00?
    # No.
    # The outer result C is 4x4.
    # C = M0 + ...
    # Each Ms is a 2x2 matrix of 2x2 blocks?
    # No. Ms is a 2x2 matrix of scalars?
    # No.
    # M_s = X_s Y_s. X_s, Y_s are 2x2 matrices of scalars.
    # So M_s is a 2x2 matrix of scalars.
    # But C is 4x4.
    # The blocks of C are 2x2 matrices.
    # C_00 = M_0[0,0] + ... ?
    # No.
    # Standard Strassen:
    # C_00 = M_1 + M_4 - M_5 + M_7
    # C_01 = M_3 + M_5
    # C_10 = M_2 + M_4
    # C_11 = M_1 - M_2 + M_3 + M_6
    # Here M_s are 2x2 **matrices**.
    # So C_00 (a 2x2 block) is the sum of 2x2 matrices M_s.
    # So the entry C[0,0] (global) is C_00[0,0] (local).
    # C_00[0,0] = M_1[0,0] + M_4[0,0] - M_5[0,0] + M_7[0,0].
    
    # So for each global position (i,k):
    # i = 2 i_b + i_l, k = 2 k_b + k_l.
    # C[i,k] = sum_s (OuterCoeff[s][i_b][k_b] * M_s[i_l, k_l])
    
    # M_s[i_l, k_l] = sum_t (InnerCoeff[t][i_l][k_l] * S_{s,t})
    
    # S_{s,t} = (sum_{j} U_{i,j,s,t} A[i,j]) * (sum_{j} V_{j,k,s,t} B[j,k])
    
    # So U[i,j, 7s+t] += OuterCoeff? No.
    # U[i,j, r] is the coefficient of A[i,j] in the first factor of the r-th product.
    # The r-th product is S_{s,t}.
    # S_{s,t} = X_t * Y_t.
    # X_t is a linear combo of X entries.
    # X entries are linear combos of A.
    
    # So for each s, t:
    # 1. Compute X_vec: 4x4 matrix of coeffs for A.
    #    X_vec[i,j] = sum_{u,v} InnerXCoeff[t][u][v] * X_coeffs[s][u][v][i][j]
    # 2. Compute Y_vec: 4x4 matrix of coeffs for B.
    #    Y_vec[j,k] = sum_{u,v} InnerYCoeff[t][u][v] * Y_coeffs[s][u][v][j][k]
    # 3. For each i, j: U[i,j, 7s+t] = X_vec[i,j]
    # 4. For each j, k: V[j,k, 7s+t] = Y_vec[j,k]
    # 5. For each k, i: W[k,i, 7s+t] = sum_{i_b,k_b} OuterW[s][i_b][k_b] * InnerW[t][i_l][k_l]
    #    Wait, W is indexed by k*n+i.
    #    The contribution to C[i,k] from S_{s,t} is:
    #    sum_{i_b,k_b} OuterW[s][i_b][k_b] * InnerW[t][i_b][k_b] ?
    #    No.
    #    C[i,k] = sum_{i_b,k_b} OuterW[s][i_b][k_b] * M_s[i_l, k_l]
    #    M_s[i_l, k_l] = sum_t InnerW[t][i_l][k_l] * S_{s,t}
    #    So Coeff of S_{s,t} in C[i,k] is:
    #    OuterW[s][i_b][k_b] * InnerW[t][i_l][k_l]
    #    This must be the same for all i,j,k in the support of U,V.
    #    Actually, W[k,i, r] is the coefficient multiplying the product.
    #    W[k,i, 7s+t] should be the coefficient of C[i,k] from this term.
    #    So W[k,i, r] = OuterW[s][i_b][k_b] * InnerW[t][i_l][k_l]
    #    Note: i_b, k_b come from i,k. i_l, k_l come from i,k.
    
    # Let's implement this.
    
    # Inner Strassen Coeffs for X and Y (indices 0,1 for 2x2 matrix entries)
    # X entries: 00, 01, 10, 11 -> indices 0,1,2,3
    # Y entries: 00, 01, 10, 11 -> indices 0,1,2,3
    
    inner_x_coeffs = [
        [1, 0, 0, 1], # t=0: X00+X11
        [0, 0, 1, 1], # t=1: X10+X11
        [1, 0, 0, 0], # t=2: X00
        [0, 0, 0, 1], # t=3: X11
        [1, 1, 0, 0], # t=4: X00+X01
        [0, 0, 1, -1], # t=5: X10-X00
        [0, 1, 0, -1]  # t=6: X01-X11
    ]
    
    inner_y_coeffs = [
        [1, 0, 0, 1], # t=0: Y00+Y11
        [1, 0, 0, 0], # t=1: Y00
        [0, 1, 0, -1], # t=2: Y01-Y11
        [0, 0, 1, -1], # t=3: Y10-Y00
        [0, 0, 0, 1], # t=4: Y11
        [1, 1, 0, 0], # t=5: Y00+Y01
        [0, 0, 1, 1]  # t=6: Y10+Y11
    ]
    
    # Inner W Coeffs (contribution to Z entries 00,01,10,11)
    inner_w_coeffs = [
        [1, 0, 0, 0], # t=0: Z00+=1
        [0, 0, 1, 0], # t=1: Z10+=1
        [0, 1, 0, 0], # t=2: Z01+=1
        [1, 0, 1, 0], # t=3: Z00+=1, Z10+=1
        [-1, 1, 0, 0], # t=4: Z00-=1, Z01+=1
        [0, 0, 0, 1], # t=5: Z11+=1
        [1, 0, 0, 0]  # t=6: Z00+=1 ... Wait.
    ]
    
    # Correct Inner W:
    # Z00 = S0 + S3 - S4 + S6
    # Z01 = S2 + S4
    # Z10 = S1 + S3
    # Z11 = S0 - S1 + S2 + S5
    
    inner_w_coeffs = [
        [1, 0, 0, 0], # t=0: Z00+=1, Z11+=1? No, S0 contributes to Z00 and Z11.
        # My inner_w_coeffs list above was per t, for Z00,Z01,Z10,Z11.
        # t=0: S0. Z00+=1, Z11+=1.
        # t=1: S1. Z10+=1, Z11-=1.
        # t=2: S2. Z01+=1, Z11+=1.
        # t=3: S3. Z00+=1, Z10+=1.
        # t=4: S4. Z00-=1, Z01+=1.
        # t=5: S5. Z11+=1.
        # t=6: S6. Z00+=1.
        
        [1, 0, 0, 1], # t=0
        [0, 0, 1, -1], # t=1
        [0, 1, 0, 1], # t=2
        [1, 0, 1, 0], # t=3
        [-1, 1, 0, 0], # t=4
        [0, 0, 0, 1], # t=5
        [1, 0, 0, 0]  # t=6
    ]
    
    # Outer W Coeffs (contribution to C blocks 00,01,10,11)
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    outer_w_coeffs = [
        [1, 0, 0, 1], # s=0: C00+=1, C11+=1
        [0, 0, 1, -1], # s=1: C10+=1, C11-=1
        [0, 1, 0, 1], # s=2: C01+=1, C11+=1
        [1, 0, 1, 0], # s=3: C00+=1, C10+=1
        [-1, 1, 0, 0], # s=4: C00-=1, C01+=1
        [0, 0, 0, 1], # s=5: C11+=1
        [1, 0, 0, 0]  # s=6: C00+=1
    ]
    
    for s in range(7):
        X_coeffs, Y_coeffs = make_X_Y(s)
        for t in range(7):
            r = 7 * s + t
            # Compute X_vec (4x4)
            X_vec = [[0]*4 for _ in range(4)]
            # Map inner indices 0,1,2,3 to u,v
            # 0:0,0 1:0,1 2:1,0 3:1,1
            for idx, val in enumerate(inner_x_coeffs[t]):
                if val != 0:
                    u = idx // 2
                    v = idx % 2
                    for i in range(4):
                        for j in range(4):
                            X_vec[i][j] += val * X_coeffs[u][v][i][j]
            
            # Compute Y_vec (4x4)
            Y_vec = [[0]*4 for _ in range(4)]
            for idx, val in enumerate(inner_y_coeffs[t]):
                if val != 0:
                    u = idx // 2
                    v = idx % 2
                    for j in range(4):
                        for k in range(4):
                            Y_vec[j][k] += val * Y_coeffs[u][v][j][k]
            
            # Fill U, V
            for i in range(4):
                for j in range(4):
                    U[i*4+j][r] = X_vec[i][j]
            
            for j in range(4):
                for k in range(4):
                    V[j*4+k][r] = Y_vec[j][k]
            
            # Fill W
            for i in range(4):
                i_b = i // 2
                i_l = i % 2
                for k in range(4):
                    k_b = k // 2
                    k_l = k % 2
                    # W index: k*n + i
                    w_val = 0
                    # Sum over s,t contributions?
                    # No, for this specific r (s,t), the coeff is:
                    # OuterW[s][i_b][k_b] * InnerW[t][i_l][k_l]
                    w_val = outer_w_coeffs[s][i_b*2+k_b] * inner_w_coeffs[t][i_l*2+k_l]
                    W[k*4+i][r] = w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
