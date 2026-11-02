"""
Matrix multiplication 4x4x8 using rank-98 Strassen-based decomposition.

The 4x4x8 multiplication is split into two 4x4x4 multiplications.
Each 4x4x4 uses Strassen's algorithm, giving rank 49 per block.
Total rank: 98 (vs schoolbook 128).

Convention: W[k*n + i, r] for the output tensor factor.
"""

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # We'll build rank-49 decompositions for 4x4x4, then combine two of them.
    # 
    # Strassen for 2x2x2 has rank 7.
    # For 4x4x4, split into 2x2 blocks, use Strassen on the 2x2 block matrix,
    # which gives 7 sub-problems each of size 2x2x2, each of rank 7.
    # Total rank = 7 * 7 = 49.
    
    # Indices: A is n x m = 4x4, B is m x p = 4x4, C is p x n = 4x4
    # Split indices into high/low: index i = i_h * 2 + i_l where i_h, i_l in {0,1}
    # A[i, j]: i = 2*i_h + i_l, j = 2*j_h + j_l
    # B[j, k]: j = 2*j_h + j_l, k = 2*k_h + k_l
    # C[k, i]: k = 2*k_h + k_l, i = 2*i_h + i_l
    
    # Strassen 2x2x2:
    # For block matrices A_b (2x2), B_b (2x2), C_b (2x2):
    # C[0,0] = M1 - M2 - M4 + M5
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M5 - M6
    # where
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # For 4x4x4:
    # A has blocks A00, A01, A10, A11 each 2x2
    # B has blocks B00, B01, B10, B11 each 2x2
    # C has blocks C00, C01, C10, C11 each 2x2
    # 
    # C00 = M1 - M2 - M4 + M5  where each M is 2x2x2 product
    # etc.
    
    # Let's enumerate the 7 Strassen sub-problems at the top level:
    # Each sub-problem is a 2x2x2 multiplication of specific linear combinations.
    
    # Top-level Strassen combinations (operating on 2x2 blocks):
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Output combinations:
    # C00 = M1 - M2 - M4 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    # For each top-level M_s (s=1..7), we need to compute a 2x2x2 product.
    # Each such 2x2x2 product uses rank 7 (Strassen at the lower level).
    
    # Let's define the 7 lower-level Strassen combinations:
    # For 2x2x2 with matrices X(2x2), Y(2x2), Z(2x2):
    # Z[0,0] = N1 - N2 - N4 + N5
    # Z[0,1] = N3 + N5
    # Z[1,0] = N2 + N4
    # Z[1,1] = N1 - N3 + N5 - N6
    # where
    # N1 = (X00 + X11)(Y00 + Y11)
    # N2 = (X10 + X11) Y00
    # N3 = X00 (Y01 - Y11)
    # N4 = X11 (Y10 - Y00)
    # N5 = (X00 + X01) Y11
    # N6 = (X10 - X00)(Y00 + Y01)
    # N7 = (X01 - X11)(Y10 + Y11)
    
    # So the full decomposition has 7 * 7 = 49 scalar multiplications.
    
    # Now let's build the U, V, W matrices for one 4x4x4 block.
    # U has shape (n*m, rank) = (16, 49)
    # V has shape (m*p, rank) = (16, 49)
    # W has shape (p*n, rank) = (16, 49)
    
    # Indexing for 4x4 matrices:
    # A[i, j] where i,j in 0..3
    # B[j, k] where j,k in 0..3
    # C[k, i] where k,i in 0..3 (note: W convention is W[k*n+i, r])
    
    # Let's define the 49 rank-1 terms.
    # Each term corresponds to a (s, t) pair where s is the top-level Strassen index (0..6)
    # and t is the lower-level Strassen index (0..6).
    
    # Top-level definitions (block level, 2x2 blocks):
    # For each s in 0..6, we define:
    # - The linear combination of A blocks to form X_s (2x2)
    # - The linear combination of B blocks to form Y_s (2x2)
    # - The contribution to C blocks from Z_s (2x2)
    
    # Let's index blocks: block (bh, bl) where bh, bl in {0,1}
    # Block A_bh_bl has rows 2*bh..2*bh+1, cols 2*bl..2*bl+1
    # Similarly for B and C.
    
    # Top-level combinations (coefficients for A blocks and B blocks):
    # s=0: X = A00 + A11, Y = B00 + B11
    # s=1: X = A10 + A11, Y = B00
    # s=2: X = A00, Y = B01 - B11
    # s=3: X = A11, Y = B10 - B00
    # s=4: X = A00 + A01, Y = B11
    # s=5: X = A10 - A00, Y = B00 + B01
    # s=6: X = A01 - A11, Y = B10 + B11
    
    # Output combinations (coefficients for C blocks from Z_s):
    # C00: [1, -1, 0, -1, 1, 0, 0] (for s=0..6)
    # C01: [0, 0, 1, 0, 1, 0, 0]
    # C10: [0, 1, 0, 1, 0, 0, 0]
    # C11: [1, 0, -1, 0, 1, -1, 0]
    
    # Wait, let me re-check:
    # C00 = M1 - M2 - M4 + M5 -> s=0:1, s=1:-1, s=3:-1, s=4:1
    # C01 = M3 + M5 -> s=2:1, s=4:1
    # C10 = M2 + M4 -> s=1:1, s=3:1
    # C11 = M1 - M3 + M5 - M6 -> s=0:1, s=2:-1, s=4:1, s=5:-1
    
    # So output coefficients (for C block 00,01,10,11 from s=0..6):
    # c_out[cb][s] where cb in 0..3 (0=C00, 1=C01, 2=C10, 3=C11)
    
    # Lower-level: for each (s, t) pair, we compute a scalar product.
    # The X_s is a 2x2 matrix formed from A blocks.
    # The Y_s is a 2x2 matrix formed from B blocks.
    # The Z_s is a 2x2 matrix which contributes to C blocks.
    
    # For the lower-level Strassen, given X (2x2) and Y (2x2):
    # We form 7 intermediate products N1..N7, each a scalar.
    # Then Z is formed from N1..N7.
    
    # Let's define the lower-level combinations:
    # For X (2x2) with elements X[0,0], X[0,1], X[1,0], X[1,1]
    # and Y (2x2) with elements Y[0,0], Y[0,1], Y[1,0], Y[1,1]:
    
    # N1 = (X00+X11)(Y00+Y11)
    # N2 = (X10+X11)Y00
    # N3 = X00(Y01-Y11)
    # N4 = X11(Y10-Y00)
    # N5 = (X00+X01)Y11
    # N6 = (X10-X00)(Y00+Y01)
    # N7 = (X01-X11)(Y10+Y11)
    
    # Z[0,0] = N1 - N2 - N4 + N5
    # Z[0,1] = N3 + N5
    # Z[1,0] = N2 + N4
    # Z[1,1] = N1 - N3 + N5 - N6
    
    # Now, X_s is formed from A blocks. Let's be more explicit.
    # A has 4 blocks: A00, A01, A10, A11, each 2x2.
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    # For s=0: X = A00 + A11
    #   X[0,0] = A[0,0] + A[2,2]
    #   X[0,1] = A[0,1] + A[2,3]
    #   X[1,0] = A[1,0] + A[3,2]
    #   X[1,1] = A[1,1] + A[3,3]
    
    # This is getting complex. Let me just build it programmatically.
    
    def build_4x4x4():
        """Build rank-49 decomposition for 4x4x4 multiplication."""
        rank_444 = 49
        U444 = [[0] * rank_444 for _ in range(16)]
        V444 = [[0] * rank_444 for _ in range(16)]
        W444 = [[0] * rank_444 for _ in range(16)]
        
        # Top-level Strassen combinations
        # s in 0..6
        # For each s, define:
        #   a_coeffs: dict from (bh, bl) to coefficient for A block (bh, bl)
        #   b_coeffs: dict from (bh, bl) to coefficient for B block (bh, bl)
        #   c_coeffs: dict from (cbh, cbl) to coefficient for C block contribution
        
        # Actually, let me think differently.
        # 
        # The tensor for 4x4x4 can be written as sum of 49 rank-1 terms.
        # Each rank-1 term is u ⊗ v ⊗ w where u is in R^16 (for A), v in R^16 (for B), w in R^16 (for C).
        # 
        # u[i*m + j] = coefficient for A[i,j]
        # v[j*p + k] = coefficient for B[j,k]
        # w[k*n + i] = coefficient for C[k,i] (with n=4, p=4)
        
        # Let me use the block structure.
        # A[i,j] with i = 2*ih + il, j = 2*jh + jl, where ih,jh,il,jl in {0,1}
        # B[j,k] with j = 2*jh + jl, k = 2*kh + kl
        # C[k,i] with k = 2*kh + kl, i = 2*ih + il
        
        # Top-level: for s in 0..6:
        #   X_s is a 2x2 matrix, element (ih, jl) is a linear combo of A elements
        #   Y_s is a 2x2 matrix, element (jh, kh) is a linear combo of B elements
        #   Z_s contributes to C blocks
        
        # Let me define the top-level combos explicitly:
        
        # s=0: X = A00 + A11, Y = B00 + B11
        #   X[ih, jl] = sum over A elements where (ih, jl) matches
        #   A00 has ih=0, jl=0; A11 has ih=1, jl=1
        #   So X[0,0] = A[0,0] + A[2,2] = A[ih=0,jh=0,il=0,jl=0] + A[ih=1,jh=1,il=0,jl=0]... 
        #   Wait, A is indexed by row i and col j.
        #   A[i,j] = A[ih, il, jh, jl] where i=2*ih+il, j=2*jh+jl
        #   A00: ih=0, jh=0, il,jl in {0,1}
        #   A11: ih=1, jh=1, il,jl in {0,1}
        
        # So X_s[ih, jl] involves A[2*ih+il, 2*jh_A+jl] where jh_A depends on s.
        
        # This is getting too complex to do by hand. Let me just hardcode the 49 terms.
        
        # Alternative approach: use known rank-49 decomposition for 4x4x4.
        # 
        # Actually, let me just implement the Strassen recursion properly.
        
        # For 4x4x4:
        # Split A into 4 2x2 blocks: A[00], A[01], A[10], A[11]
        # Split B into 4 2x2 blocks: B[00], B[01], B[10], B[11]
        # C = A*B, split into 4 2x2 blocks: C[00], C[01], C[10], C[11]
        
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M2 - M4 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 - M6
        
        # Now each M_s is a 2x2x2 multiplication. We use Strassen for each.
        
        # For a 2x2x2 multiplication X(2x2) * Y(2x2) = Z(2x2):
        # N1 = (X00+X11)(Y00+Y11)
        # N2 = (X10+X11)Y00
        # N3 = X00(Y01-Y11)
        # N4 = X11(Y10-Y00)
        # N5 = (X00+X01)Y11
        # N6 = (X10-X00)(Y00+Y01)
        # N7 = (X01-X11)(Y10+Y11)
        
        # Z00 = N1 - N2 - N4 + N5
        # Z01 = N3 + N5
        # Z10 = N2 + N4
        # Z11 = N1 - N3 + N5 - N6
        
        # Now let's build the rank-49 decomposition.
        # Each of the 7 M_s computations uses 7 scalar multiplications.
        # Total: 49 scalar multiplications.
        
        # For each (s, t) where s in 0..6 (top-level) and t in 0..6 (bottom-level):
        # We need to find the coefficients for A, B, C.
        
        # Let me define helper functions to get the X_s and Y_s blocks.
        
        # A[i,j] where i = 2*ih + il, j = 2*jh + jl
        # Block A[bh, bl] has elements A[2*bh+il, 2*bl+jl] for il,jl in {0,1}
        
        # Top-level combos:
        # s=0: X = A00 + A11 -> X[ih, jl] = sum of A elements
        #   X[0,0] = A[0,0] + A[2,2]
        #   X[0,1] = A[0,1] + A[2,3]
        #   X[1,0] = A[1,0] + A[3,2]
        #   X[1,1] = A[1,1] + A[3,3]
        
        #   Y = B00 + B11 -> Y[jh, kh] = sum of B elements
        #   Y[0,0] = B[0,0] + B[2,2]
        #   Y[0,1] = B[0,1] + B[2,3]
        #   Y[1,0] = B[1,0] + B[3,2]
        #   Y[1,1] = B[1,1] + B[3,3]
        
        # s=1: X = A10 + A11
        #   X[0,0] = A[2,0] + A[2,2]
        #   X[0,1] = A[2,1] + A[2,3]
        #   X[1,0] = A[3,0] + A[3,2]
        #   X[1,1] = A[3,1] + A[3,3]
        
        #   Y = B00
        #   Y[0,0] = B[0,0]
        #   Y[0,1] = B[0,1]
        #   Y[1,0] = B[1,0]
        #   Y[1,1] = B[1,1]
        
        # etc.
        
        # This is very tedious to hardcode. Let me write it programmatically.
        
        # Define the top-level A and B block combinations:
        # a_combo[s][bh][bl] = coefficient for A block (bh, bl) in X_s
        # b_combo[s][bh][bl] = coefficient for B block (bh, bl) in Y_s
        
        a_combo = {}
        b_combo = {}
        
        # s=0: X = A00 + A11, Y = B00 + B11
        a_combo[0] = {(0,0): 1, (1,1): 1}
        b_combo[0] = {(0,0): 1, (1,1): 1}
        
        # s=1: X = A10 + A11, Y = B00
        a_combo[1] = {(1,0): 1, (1,1): 1}
        b_combo[1] = {(0,0): 1}
        
        # s=2: X = A00, Y = B01 - B11
        a_combo[2] = {(0,0): 1}
        b_combo[2] = {(0,1): 1, (1,1): -1}
        
        # s=3: X = A11, Y = B10 - B00
        a_combo[3] = {(1,1): 1}
        b_combo[3] = {(1,0): 1, (0,0): -1}
        
        # s=4: X = A00 + A01, Y = B11
        a_combo[4] = {(0,0): 1, (0,1): 1}
        b_combo[4] = {(1,1): 1}
        
        # s=5: X = A10 - A00, Y = B00 + B01
        a_combo[5] = {(1,0): 1, (0,0): -1}
        b_combo[5] = {(0,0): 1, (0,1): 1}
        
        # s=6: X = A01 - A11, Y = B10 + B11
        a_combo[6] = {(0,1): 1, (1,1): -1}
        b_combo[6] = {(1,0): 1, (1,1): 1}
        
        # C output combinations:
        # c_out[cb][s] = coefficient for Z_s in C block cb
        # cb: 0=C00, 1=C01, 2=C10, 3=C11
        c_out = {
            0: {0: 1, 1: -1, 3: -1, 4: 1},  # C00 = M1 - M2 - M4 + M5
            1: {2: 1, 4: 1},                  # C01 = M3 + M5
            2: {1: 1, 3: 1},                  # C10 = M2 + M4
            3: {0: 1, 2: -1, 4: 1, 5: -1}    # C11 = M1 - M3 + M5 - M6
        }
        
        # Now for each (s, t), we compute the rank-1 term.
        # t in 0..6 corresponds to N1..N7 in the lower-level Strassen.
        
        # Lower-level Strassen combos:
        # N1 = (X00+X11)(Y00+Y11)
        # N2 = (X10+X11)Y00
        # N3 = X00(Y01-Y11)
        # N4 = X11(Y10-Y00)
        # N5 = (X00+X01)Y11
        # N6 = (X10-X00)(Y00+Y01)
        # N7 = (X01-X11)(Y10+Y11)
        
        # X[ih, jl] and Y[jh, kh]
        # N_t is a scalar product of two linear combos.
        
        # For N1: X part = X00 + X11, Y part = Y00 + Y11
        #   X00 = X[0,0], X11 = X[1,1]
        #   Y00 = Y[0,0], Y11 = Y[1,1]
        
        # Let me define x_combo[t][ih, jl] and y_combo[t][jh, kh]:
        
        x_combo = {}
        y_combo = {}
        
        # t=0 (N1): X = X00 + X11, Y = Y00 + Y11
        x_combo[0] = {(0,0): 1, (1,1): 1}
        y_combo[0] = {(0,0): 1, (1,1): 1}
        
        # t=1 (N2): X = X10 + X11, Y = Y00
        x_combo[1] = {(1,0): 1, (1,1): 1}
        y_combo[1] = {(0,0): 1}
        
        # t=2 (N3): X = X00, Y = Y01 - Y11
        x_combo[2] = {(0,0): 1}
        y_combo[2] = {(0,1): 1, (1,1): -1}
        
        # t=3 (N4): X = X11, Y = Y10 - Y00
        x_combo[3] = {(1,1): 1}
        y_combo[3] = {(1,0): 1, (0,0): -1}
        
        # t=4 (N5): X = X00 + X01, Y = Y11
        x_combo[4] = {(0,0): 1, (0,1): 1}
        y_combo[4] = {(1,1): 1}
        
        # t=5 (N6): X = X10 - X00, Y = Y00 + Y01
        x_combo[5] = {(1,0): 1, (0,0): -1}
        y_combo[5] = {(0,0): 1, (0,1): 1}
        
        # t=6 (N7): X = X01 - X11, Y = Y10 + Y11
        x_combo[6] = {(0,1): 1, (1,1): -1}
        y_combo[6] = {(1,0): 1, (1,1): 1}
        
        # Z output:
        # Z00 = N1 - N2 - N4 + N5 -> t=0:1, t=1:-1, t=3:-1, t=4:1
        # Z01 = N3 + N5 -> t=2:1, t=4:1
        # Z10 = N2 + N4 -> t=1:1, t=3:1
        # Z11 = N1 - N3 + N5 - N6 -> t=0:1, t=2:-1, t=4:1, t=5:-1
        z_out = {
            (0,0): {0: 1, 1: -1, 3: -1, 4: 1},
            (0,1): {2: 1, 4: 1},
            (1,0): {1: 1, 3: 1},
            (1,1): {0: 1, 2: -1, 4: 1, 5: -1}
        }
        
        # Now build U, V, W for 4x4x4.
        # For each (s, t) in 0..6 x 0..6:
        #   col = s * 7 + t
        #   U[2*ih+il, 2*jh+jl + col] += ... wait, U is indexed by [i*m + j]
        #   U[i*4 + j, col] where i = row of A, j = col of A
        
        for s in range(7):
            for t in range(7):
                col = s * 7 + t
                
                # A coefficients:
                # X_s[ih, jl] = sum over A blocks
                # X_s[ih, jl] = sum_{(ab_h, ab_l)} a_combo[s][(ab_h, ab_l)] * A[2*ab_h+ih, 2*ab_l+jl]
                # So A[i,j] where i = 2*ab_h + ih, j = 2*ab_l + jl
                # U[i*4 + j, col] += a_combo[s][(ab_h, ab_l)] * x_combo[t][(ih, jl)]
                
                for ab_h in range(2):
                    for ab_l in range(2):
                        if (ab_h, ab_l) in a_combo[s]:
                            coeff_a = a_combo[s][(ab_h, ab_l)]
                            for ih in range(2):
                                for jl in range(2):
                                    if (ih, jl) in x_combo[t]:
                                        coeff_x = x_combo[t][(ih, jl)]
                                        i = 2 * ab_h + ih
                                        j = 2 * ab_l + jl
                                        U444[i * 4 + j][col] += coeff_a * coeff_x
                
                # B coefficients:
                # Y_s[jh, kh] = sum over B blocks
                # Y_s[jh, kh] = sum_{(bb_h, bb_l)} b_combo[s][(bb_h, bb_l)] * B[2*bb_h+jl, 2*bb_l+kh]
                # Wait, B[j,k] where j = 2*bb_h + jl, k = 2*bb_l + kh
                # V[j*4 + k, col] += b_combo[s][(bb_h, bb_l)] * y_combo[t][(jl, kh)]
                # Wait, Y_s is indexed by (jh, kh) but in the lower-level, 
                # Y[jh, kh] corresponds to B elements where j = 2*bb_h + jh? No...
                
                # Let me re-think. 
                # In the top-level, B is split into blocks B[bh, bl] where 
                # B[bh, bl] has elements B[2*bh+il, 2*bl+jl] for il,jl in {0,1}.
                # Wait, B is m x p = 4 x 4. B[j,k].
                # Block B[bh, bl]: rows 2*bh..2*bh+1, cols 2*bl..2*bl+1.
                # B[2*bh+il, 2*bl+jl].
                
                # In the lower-level Strassen, Y is a 2x2 matrix.
                # Y[ih, jl] where these are the row/col indices of the 2x2 block.
                # But Y_s is formed from B blocks.
                # Y_s[ih, jl] = sum_{(bb_h, bb_l)} b_combo[s][(bb_h, bb_l)] * B_block[bb_h, bb_l][ih, jl]
                # B_block[bb_h, bb_l][ih, jl] = B[2*bb_h + ih, 2*bb_l + jl]
                
                # So B[j,k] where j = 2*bb_h + ih, k = 2*bb_l + jl
                # V[j*4 + k, col] += b_combo[s][(bb_h, bb_l)] * y_combo[t][(ih, jl)]
                
                for bb_h in range(2):
                    for bb_l in range(2):
                        if (bb_h, bb_l) in b_combo[s]:
                            coeff_b = b_combo[s][(bb_h, bb_l)]
                            for ih in range(2):
                                for jl in range(2):
                                    if (ih, jl) in y_combo[t]:
                                        coeff_y = y_combo[t][(ih, jl)]
                                        j = 2 * bb_h + ih
                                        k = 2 * bb_l + jl
                                        V444[j * 4 + k][col] += coeff_b * coeff_y
                
                # C coefficients:
                # Z_s contributes to C blocks.
                # Z_s[ih, jl] (2x2 block) contributes to C blocks.
                # C block (cbh, cbl) where C[2*cbh+kh, 2*cbl+il]
                # Z_s is a 2x2 matrix with elements Z_s[ih, jl]
                # Z_s[ih, jl] = sum_t z_out[(cbh, cbl)][t] * N_t for the appropriate (cbh, cbl)
                
                # Actually, Z_s[ih, jl] is the (ih, jl) element of the 2x2 result Z_s.
                # Z_s[ih, jl] = sum over t of z_out[(ih, jl)][t] * N_t
                # And N_t contributes to C block.
                
                # C[k, i] where k = 2*kh + kl, i = 2*ih + il
                # W[k*4 + i, col] += ...
                
                # The Z_s block is a 2x2 matrix. Z_s[il, ih]... wait.
                # In the 2x2x2 multiplication, Z = X * Y.
                # Z[il, ih] where il is row of Z (which corresponds to k in C?), ih is col of Z (which corresponds to i in C?).
                
                # Hmm, let me be more careful.
                # C[k, i] where k is from B's column (p=4), i is from A's row (n=4).
                # In the block structure:
                # C block (cbh, cbl) has elements C[2*cbh+kl, 2*cbl+il] for kl, il in {0,1}.
                # So k = 2*cbh + kl, i = 2*cbl + il.
                
                # Z_s is a 2x2 matrix. Z_s[?, ?]
                # In the 2x2x2 product, if X is 2x2 (rows indexed by one thing, cols by another)
                # and Y is 2x2, then Z = X*Y is 2x2.
                
                # For the top-level: X_s is formed from A blocks, Y_s from B blocks.
                # X_s[ih, jl] where ih indexes A's row block, jl indexes A's col block (which is B's row block).
                # Y_s[jh, kh] where jh indexes B's row block, kh indexes B's col block.
                # Z_s = X_s * Y_s, so Z_s[ih, kh] = sum_jl X_s[ih, jl] * Y_s[jl, kh]
                # Wait, that's not right either. Let me just use the indices.
                
                # Z_s is a 2x2 matrix. Its elements are Z_s[a, b] for a, b in {0,1}.
                # Z_s[a, b] contributes to C block.
                # C block (cbh, cbl) where the block is 2x2.
                # Z_s[a, b] = C block element at position (b, a)? Or (a, b)?
                
                # In standard matrix multiplication C = A * B:
                # C[i, k] = sum_j A[i, j] * B[j, k]
                # Block level: C[crow, ccol] = sum_block A[arow, bcol] * B[arow, bcol]... 
                
                # I think I'm overcomplicating this. Let me just trust the Strassen formulas and 
                # map the indices correctly.
                
                # For the 4x4x4 case:
                # C[k, i] = sum_j A[i, j] * B[j, k]
                # 
                # Block structure:
                # A[i, j] with i = 2*ih + il, j = 2*jh + jl
                # B[j, k] with j = 2*jh + jl, k = 2*kh + kl
                # C[k, i] with k = 2*kh + kl, i = 2*ih + il
                #
                # C block (kh, ih) has elements C[2*kh+kl, 2*ih+il]
                #
                # Z_s is a 2x2 matrix. Z_s[kh, ih] contributes to C block (kh, ih).
                # Z_s[kh, ih] = sum_jl X_s[ih, jl] * Y_s[jl, kh]
                #
                # But Z_s is computed via Strassen at the lower level.
                # Z_s[kh, ih] = sum_t z_out[(kh, ih)][t] * N_t
                #
                # So W[2*kh+kl, 2*ih+il, col] for kl, il in {0,1}:
                # W[(2*kh+kl)*4 + (2*ih+il), col] += z_out[(kh, ih)][t]
                
                for kh in range(2):
                    for ih in range(2):
                        if (kh, ih) in z_out:
                            coeff_z = z_out[(kh, ih)][t]
                            for kl in range(2):
                                for il in range(2):
                                    k = 2 * kh + kl
                                    i = 2 * ih + il
                                    W444[k * 4 + i][col] += coeff_z
        
        return U444, V444, W444
    
    U444, V444, W444 = build_4x4x4()
    
    # Now we need to extend this to 4x4x8.
    # B is 4x8, so we split B into two 4x4 blocks: B1 (cols 0-3) and B2 (cols 4-7).
    # C is 8x4, split into C1 (rows 0-3) and C2 (rows 4-7).
    # C1 = A * B1, C2 = A * B2.
    
    # So we use two copies of the 4x4x4 decomposition.
    # Total rank = 49 + 49 = 98.
    
    rank = 98
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]  # m*p = 4*8 = 32
    W = [[0] * rank for _ in range(32)]  # p*n = 8*4 = 32
    
    # First 49 columns: for B1 (cols 0-3 of B, rows 0-3 of C)
    for i in range(16):
        for c in range(49):
            U[i][c] = U444[i][c]
            V[i][c] = V444[i][c]
            W[i][c] = W444[i][c]
    
    # Next 49 columns: for B2 (cols 4-7 of B, rows 4-7 of C)
    for i in range(16):
        for c in range(49):
            U[i][c + 49] = U444[i][c]
            # V: B2 is cols 4-7, so V[j*8 + (k+4)] for k in 0..3
            # V444[j*4 + k] -> V[j*8 + (k+4)]
            j = i // 4
            k = i % 4
            V[j * 8 + (k + 4)][c + 49] = V444[i][c]
            # W: C2 is rows 4-7, so W[(l+4)*4 + i_c] for l in 0..3
            # W444[l*4 + i_c] -> W[(l+4)*4 + i_c]
            l = i // 4
            i_c = i % 4
            W[(l + 4) * 4 + i_c][c + 49] = W444[i][c]
    
    return {"U": U, "V": V, "W": W}
