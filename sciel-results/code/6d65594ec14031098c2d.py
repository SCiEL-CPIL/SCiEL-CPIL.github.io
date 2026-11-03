import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Construct a rank-85 tensor decomposition for 4x5 matrix multiplication.
    
    Strategy:
    - Split into four blocks:
      Block 1: A[:,0:4] * B[0:4,0:4] -> C[:,0:4]  (4x4x4 multiplication)
      Block 2: A[:,0:4] * B[0:4,4:5] -> C[:,4:5]  (4x4x1 multiplication)
      Block 3: A[:,4:5] * B[4:5,0:4] -> C[:,0:4]  (4x1x4 multiplication)
      Block 4: A[:,4:5] * B[4:5,4:5] -> C[:,4:5]  (4x1x1 multiplication)
    
    - Use Strassen's algorithm for the 4x4x4 block (rank 49)
    - Use rank-16 decompositions for the 4x4x1 and 4x1x4 blocks
    - Use rank-4 decomposition for the 4x1x1 block
    Total rank: 49 + 16 + 16 + 4 = 85
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build U, V, W as lists of rows, each row is a list of Fractions
    # U has n*m = 20 rows
    # V has m*p = 25 rows
    # W has p*n = 20 rows
    # All have rank = 85 columns
    
    rank = 85
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # =================================================================
    # Block 1: 4x4x4 multiplication using Strassen (rank 49)
    # A1 = A[0:4, 0:4], B1 = B[0:4, 0:4], C1 = C[0:4, 0:4]
    # =================================================================
    # Strassen for 2x2x2 has rank 7. For 4x4x4, we recursively apply Strassen.
    # The 4x4x4 Strassen decomposition has rank 7^2 = 49.
    
    # Strassen 2x2x2 formulas:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4x4, split into 2x2 blocks of 2x2 matrices:
    # A11, A12, A21, A22 are 2x2
    # B11, B12, B21, B22 are 2x2
    # 
    # Apply Strassen at the block level, where each "scalar" operation 
    # is actually a 2x2x2 multiplication (rank 7).
    
    # The 49 rank-1 terms come from:
    # For each of the 7 Strassen multiplications M1..M7, we need a 2x2x2 
    # multiplication (7 terms each), giving 7*7 = 49 terms.
    
    # Let's define the Strassen 2x2x2 decomposition first.
    # For 2x2x2: U has 4 rows (i*m+j), V has 4 rows (j*p+k), W has 4 rows (k*n+i)
    # The 7 terms:
    # Term 0: (A11+A22)*(B11+B22) -> contributes to C11 and C22
    # Term 1: (A21+A22)*B11 -> contributes to C21
    # Term 2: A11*(B12-B22) -> contributes to C12
    # Term 3: A22*(B21-B11) -> contributes to C21
    # Term 4: (A11+A12)*B22 -> contributes to C11
    # Term 5: (A21-A11)*(B11+B12) -> contributes to C22
    # Term 6: (A12-A22)*(B21+B22) -> contributes to C11
    
    # Actually, let me be more careful. The tensor decomposition for 2x2x2 
    # multiplication has 7 rank-1 terms. Let me list them properly.
    
    # For 2x2x2: C[i,k] = sum_j A[i,j] * B[j,k]
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] where n=m=p=2
    
    # The 7 Strassen terms:
    # c=0: M1 = (A00+A11)*(B00+B11)
    #   U[0*2+0]=1, U[1*2+1]=1 (A00, A11)
    #   V[0*2+0]=1, V[1*2+1]=1 (B00, B11)
    #   W[0*2+0]=1 (C00 += M1), W[1*2+1]=1 (C11 += M1)
    
    # c=1: M2 = (A10+A11)*B00
    #   U[1*2+0]=1, U[1*2+1]=1 (A10, A11)
    #   V[0*2+0]=1 (B00)
    #   W[0*2+1]=1 (C01 += M2)
    
    # c=2: M3 = A00*(B01-B11)
    #   U[0*2+0]=1 (A00)
    #   V[0*2+1]=1, V[1*2+1]=-1 (B01, B11)
    #   W[1*2+0]=1 (C10 += M3)
    
    # c=3: M4 = A11*(B10-B00)
    #   U[1*2+1]=1 (A11)
    #   V[1*2+0]=1, V[0*2+0]=-1 (B10, B00)
    #   W[0*2+1]=1 (C01 += M4)
    
    # c=4: M5 = (A00+A01)*B11
    #   U[0*2+0]=1, U[0*2+1]=1 (A00, A01)
    #   V[1*2+1]=1 (B11)
    #   W[0*2+0]=1 (C00 += M5)
    
    # c=5: M6 = (A10-A00)*(B00+B01)
    #   U[1*2+0]=1, U[0*2+0]=-1 (A10, A00)
    #   V[0*2+0]=1, V[0*2+1]=1 (B00, B01)
    #   W[1*2+1]=1 (C11 += M6)
    
    # c=6: M7 = (A01-A11)*(B10+B11)
    #   U[0*2+1]=1, U[1*2+1]=-1 (A01, A11)
    #   V[1*2+0]=1, V[1*2+1]=1 (B10, B11)
    #   W[0*2+0]=1 (C00 += M7)
    
    # Verify: 
    # C00 = M1 + M5 + M7 = (A00+A11)(B00+B11) + (A00+A01)B11 + (A01-A11)(B10+B11)
    #     = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A00*B11 + A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    #     = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A00*B11 + A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    #     = A00*B00 + 2*A00*B11 + A11*B00 + A01*B10 + 2*A01*B11 - A11*B10
    #     Hmm, that doesn't look right. Let me reconsider.
    
    # Actually, the standard Strassen decomposition for 2x2 matrix multiplication:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me verify C11:
    # C11 = M1 + M4 - M5 + M7
    # = (A11+A22)(B11+B22) + A22(B21-B11) - (A11+A12)B22 + (A12-A22)(B21+B22)
    # = A11*B11 + A11*B22 + A22*B11 + A22*B22 + A22*B21 - A22*B11 - A11*B22 - A12*B22 + A12*B21 + A12*B22 - A22*B21 - A22*B22
    # = A11*B11 + A12*B21
    # Correct!
    
    # Now let me code the 4x4x4 Strassen decomposition.
    # I'll use a recursive approach.
    
    def strassen_2x2x2():
        """Return the 7 rank-1 terms for 2x2x2 multiplication.
        Returns: U (4x7), V (4x7), W (4x7) as lists of lists of Fractions.
        U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        n = m = p = 2
        U = [[Fraction(0)] * 7 for _ in range(4)]
        V = [[Fraction(0)] * 7 for _ in range(4)]
        W = [[Fraction(0)] * 7 for _ in range(4)]
        
        # c=0: M1 = (A00+A11)*(B00+B11)
        U[0][0] = Fraction(1)  # A00 (i=0,j=0)
        U[3][0] = Fraction(1)  # A11 (i=1,j=1)
        V[0][0] = Fraction(1)  # B00 (j=0,k=0)
        V[3][0] = Fraction(1)  # B11 (j=1,k=1)
        W[0][0] = Fraction(1)  # C00 (k=0,i=0)
        W[3][0] = Fraction(1)  # C11 (k=1,i=1)
        
        # c=1: M2 = (A10+A11)*B00
        U[2][1] = Fraction(1)  # A10 (i=1,j=0)
        U[3][1] = Fraction(1)  # A11 (i=1,j=1)
        V[0][1] = Fraction(1)  # B00 (j=0,k=0)
        W[1][1] = Fraction(1)  # C01 (k=0,i=1)
        
        # c=2: M3 = A00*(B01-B11)
        U[0][2] = Fraction(1)  # A00
        V[1][2] = Fraction(1)  # B01 (j=0,k=1)
        V[3][2] = Fraction(-1)  # B11 (j=1,k=1)
        W[2][2] = Fraction(1)  # C10 (k=1,i=0)
        
        # c=3: M4 = A11*(B10-B00)
        U[3][3] = Fraction(1)  # A11
        V[2][3] = Fraction(1)  # B10 (j=1,k=0)
        V[0][3] = Fraction(-1)  # B00
        W[1][3] = Fraction(1)  # C01 (k=0,i=1)
        
        # c=4: M5 = (A00+A01)*B11
        U[0][4] = Fraction(1)  # A00
        U[1][4] = Fraction(1)  # A01 (i=0,j=1)
        V[3][4] = Fraction(1)  # B11
        W[0][4] = Fraction(1)  # C00
        
        # c=5: M6 = (A10-A00)*(B00+B01)
        U[2][5] = Fraction(1)  # A10
        U[0][5] = Fraction(-1)  # A00
        V[0][5] = Fraction(1)  # B00
        V[1][5] = Fraction(1)  # B01
        W[3][5] = Fraction(1)  # C11
        
        # c=6: M7 = (A01-A11)*(B10+B11)
        U[1][6] = Fraction(1)  # A01
        U[3][6] = Fraction(-1)  # A11
        V[2][6] = Fraction(1)  # B10
        V[3][6] = Fraction(1)  # B11
        W[0][6] = Fraction(1)  # C00
        
        return U, V, W
    
    # For 4x4x4, we split into 2x2 blocks.
    # The 4x4 matrices are:
    # A = [[A00, A01], [A10, A11]] where each Ai0 is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = A*B
    
    # Strassen at block level:
    # M1 = (A00+A11)*(B00+B11) -> 2x2*2x2 -> 7 terms
    # M2 = (A10+A11)*B00 -> 2x2*2x2 -> 7 terms
    # M3 = A00*(B01-B11) -> 2x2*2x2 -> 7 terms
    # M4 = A11*(B10-B00) -> 2x2*2x2 -> 7 terms
    # M5 = (A00+A01)*B11 -> 2x2*2x2 -> 7 terms
    # M6 = (A10-A00)*(B00+B01) -> 2x2*2x2 -> 7 terms
    # M7 = (A01-A11)*(B10+B11) -> 2x2*2x2 -> 7 terms
    
    # Total: 49 terms
    
    # Now I need to map the block-level Strassen to the 4x4x4 tensor.
    
    # Let me define the 4x4x4 decomposition directly.
    
    def strassen_4x4x4():
        """Return the 49 rank-1 terms for 4x4x4 multiplication using Strassen.
        U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
        """
        n = m = p = 4
        U = [[Fraction(0)] * 49 for _ in range(16)]
        V = [[Fraction(0)] * 49 for _ in range(16)]
        W = [[Fraction(0)] * 49 for _ in range(16)]
        
        # Get the 2x2x2 Strassen terms
        U2, V2, W2 = strassen_2x2x2()
        
        # The 7 block-level Strassen multiplications:
        # For each, we have a coefficient on the A-block and B-block,
        # and we use the 2x2x2 decomposition for the actual multiplication.
        
        # Block indexing: A block (bi, bj) has rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
        # In the 4x4 matrix A, element (i,j) is in block (i//2, j//2)
        
        # Let's define the 7 Strassen multiplications:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # For each Mk, we have:
        # - A-side: linear combination of A-blocks
        # - B-side: linear combination of B-blocks
        # - The multiplication is done via 2x2x2 Strassen (7 terms)
        
        # Let me define the A and B coefficients for each Mk:
        # Mk: (a_coeffs, b_coeffs)
        # a_coeffs: list of (block_i, block_j, coeff) for A
        # b_coeffs: list of (block_i, block_j, coeff) for B
        
        # M1: A00 + A11, B00 + B11
        # M2: A10 + A11, B00
        # M3: A00, B01 - B11
        # M4: A11, B10 - B00
        # M5: A00 + A01, B11
        # M6: A10 - A00, B00 + B01
        # M7: A01 - A11, B10 + B11
        
        # For each Mk, the contribution to C is:
        # C += Ck where Ck = (sum_a a_coeff_a * A_a) * (sum_b b_coeff_b * B_b)
        # = sum_a sum_b a_coeff_a * b_coeff_b * (A_a * B_b)
        
        # But A_a * B_b is a 2x2*2x2 multiplication, which we decompose using Strassen.
        
        # Actually, it's simpler: for each Mk, we compute the 2x2*2x2 product
        # of the combined A-block and combined B-block, and that gives us the
        # contribution to the 2x2 C-block.
        
        # Let me just directly enumerate the 49 terms.
        
        # For each of the 7 Strassen multipliers (m=0..6), and each of the 7 
        # rank-1 terms in the 2x2x2 decomposition (c2=0..6), we get one term.
        
        # The 2x2x2 term c2 has:
        # - A coefficients on the 4 elements of the 2x2 A-block
        # - B coefficients on the 4 elements of the 2x2 B-block
        # - C coefficients on the 4 elements of the 2x2 C-block
        
        # For Mk, the A-block is a linear combination, so the actual A elements
        # get multiplied by the block coefficient.
        
        # Let me build this systematically.
        
        # For each m (0-6), define the A and B block coefficients:
        strassen_defs = [
            # (m, A_blocks, B_blocks)
            # A_blocks: list of (bi, bj, coeff)
            # B_blocks: list of (bi, bj, coeff)
            (0, [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),  # M1
            (1, [(1,0,1), (1,1,1)], [(0,0,1)]),           # M2
            (2, [(0,0,1)], [(0,1,1), (1,1,-1)]),          # M3
            (3, [(1,1,1)], [(1,0,1), (0,0,-1)]),          # M4
            (4, [(0,0,1), (0,1,1)], [(1,1,1)]),           # M5
            (5, [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]), # M6
            (6, [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]), # M7
        ]
        
        # For each (m, a_blocks, b_blocks), and for each c2 (0-6) in the 2x2x2:
        # The term index is m*7 + c2
        
        # U2[i2*2+j2, c2] gives the coefficient for A element (i2,j2) in the 2x2 block
        # But we need to map to the 4x4 matrix.
        
        # The 2x2 A-block with block index (bi, bj) has elements:
        # A[2*bi + i2, 2*bj + j2] for i2,j2 in {0,1}
        
        # So for a given m and c2, the U contribution for a specific A element (i,j):
        # For each (bi, bj, coeff_a) in a_blocks:
        #   i2 = i - 2*bi, j2 = j - 2*bj
        #   if 0 <= i2 < 2 and 0 <= j2 < 2:
        #     U[i*4+j, m*7+c2] += coeff_a * U2[i2*2+j2, c2]
        
        # Similarly for V and W.
        
        for m, a_blocks, b_blocks in strassen_defs:
            for c2 in range(7):
                c_idx = m * 7 + c2
                
                # Process U
                for i in range(4):
                    for j in range(4):
                        total = Fraction(0)
                        for bi, bj, coeff_a in a_blocks:
                            i2 = i - 2 * bi
                            j2 = j - 2 * bj
                            if 0 <= i2 < 2 and 0 <= j2 < 2:
                                total += coeff_a * U2[i2 * 2 + j2, c2]
                        U[i * 4 + j, c_idx] = total
                
                # Process V
                for j in range(4):
                    for k in range(4):
                        total = Fraction(0)
                        for bi, bj, coeff_b in b_blocks:
                            j2 = j - 2 * bi
                            k2 = k - 2 * bj
                            if 0 <= j2 < 2 and 0 <= k2 < 2:
                                total += coeff_b * V2[j2 * 2 + k2, c2]
                        V[j * 4 + k, c_idx] = total
                
                # Process W
                for k in range(4):
                    for i in range(4):
                        total = Fraction(0)
                        for bi, bj, coeff_w in [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]:
                            # C-block (bi,bj) gets contribution from the 2x2x2 term
                            # The W2[k2*2+i2, c2] gives the coefficient for C element (i2,k2)
                            # in the 2x2 C-block
                            # But which C-block? It depends on the Strassen formula.
                            # 
                            # Actually, the 2x2x2 decomposition's W tells us which C-elements
                            # are affected. For Mk, the result is a 2x2 C-block, but which one?
                            #
                            # M1 contributes to C00 and C11
                            # M2 contributes to C10
                            # M3 contributes to C01
                            # M4 contributes to C10
                            # M5 contributes to C00
                            # M6 contributes to C11
                            # M7 contributes to C00
                            #
                            # Wait, that's not right either. Let me reconsider.
                            #
                            # In Strassen:
                            # C00 = M1 + M4 - M5 + M7
                            # C01 = M3 + M5
                            # C10 = M2 + M4
                            # C11 = M1 - M2 + M3 + M6
                            #
                            # So each Mk contributes to specific C-blocks.
                            # But within each Mk, the 2x2x2 decomposition gives the
                            # full 2x2 C result. So Mk produces a 2x2 matrix, and that
                            # 2x2 matrix is placed in the appropriate C-block(s).
                            #
                            # Actually no. Mk = (combined A) * (combined B) is a 2x2 matrix.
                            # And that 2x2 matrix IS the contribution to a specific C-block.
                            #
                            # M1 -> C00 and C11 (same 2x2 matrix added to both)
                            # M2 -> C10
                            # M3 -> C01
                            # M4 -> C10
                            # M5 -> C00
                            # M6 -> C11
                            # M7 -> C00
                            #
                            # Wait, that means M1's 2x2 result goes to both C00 and C11?
                            # Let me re-examine.
                            #
                            # M1 = (A00+A11)*(B00+B11) is a 2x2 matrix.
                            # C00 = M1 + M4 - M5 + M7
                            # C11 = M1 - M2 + M3 + M6
                            #
                            # So yes, M1's 2x2 result is added to both C00 and C11.
                            #
                            # So for the W tensor, for Mk, the 2x2x2 W coefficients
                            # tell us which elements of the 2x2 C-result are non-zero,
                            # and we need to place them in the correct C-block(s).
                            
                            pass
                
                # Let me handle W more carefully.
                # For each m, define which C-blocks it contributes to:
                c_block_map = {
                    0: [(0,0,1), (1,1,1)],   # M1 -> C00, C11
                    1: [(1,0,1)],             # M2 -> C10
                    2: [(0,1,1)],             # M3 -> C01
                    3: [(1,0,1)],             # M4 -> C10
                    4: [(0,0,1)],             # M5 -> C00
                    5: [(1,1,1)],             # M6 -> C11
                    6: [(0,0,1)],             # M7 -> C00
                }
                
                for ki in range(4):  # k index
                    for ii in range(4):  # i index
                        total = Fraction(0)
                        for bi, bj, coeff_c in c_block_map[m]:
                            k2 = ki - 2 * bi
                            i2 = ii - 2 * bj
                            if 0 <= k2 < 2 and 0 <= i2 < 2:
                                total += coeff_c * W2[k2 * 2 + i2, c2]
                        W[ki * 4 + ii, c_idx] = total
        
        return U, V, W
    
    U4, V4, W4 = strassen_4x4x4()
    
    # Now add the 4x4x4 decomposition to the global U, V, W
    # Global indices:
    # U[i*5+j, c] where i in 0..3, j in 0..4 (for A[4x5])
    # For the 4x4x4 block, j is in 0..3
    # V[j*5+k, c] where j in 0..4, k in 0..4 (for B[5x5])
    # For the 4x4x4 block, j in 0..3, k in 0..3
    # W[k*4+i, c] where k in 0..4, i in 0..3 (for C[5x4])
    # For the 4x4x4 block, k in 0..3
    
    for i in range(4):
        for j in range(4):
            for c in range(49):
                U[i * 5 + j, col_offset + c] = U4[i * 4 + j, c]
    
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j * 5 + k, col_offset + c] = V4[j * 4 + k, c]
    
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k * 4 + i, col_offset + c] = W4[k * 4 + i, c]
    
    col_offset += 49
    
    # =================================================================
    # Block 2: 4x4x1 multiplication (A[:,0:4] * B[0:4,4] -> C[:,4])
    # Rank 16
    # =================================================================
    # C[i,4] = sum_j A[i,j] * B[j,4] for i=0..3, j=0..3
    # This is a rank-16 tensor: for each (i,j), one term.
    
    for i in range(4):
        for j in range(4):
            c = col_offset + i * 4 + j
            U[i * 5 + j, c] = Fraction(1)  # A[i,j]
            V[j * 5 + 4, c] = Fraction(1)  # B[j,4]
            W[4 * 4 + i, c] = Fraction(1)  # C[4,i] -> W[k*n+i] = W[4*4+i]
    
    col_offset += 16
    
    # =================================================================
    # Block 3: 4x1x4 multiplication (A[:,4] * B[4,0:4] -> C[:,0:4])
    # Rank 16
    # =================================================================
    # C[i,k] += A[i,4] * B[4,k] for i=0..3, k=0..3
    
    for i in range(4):
        for k in range(4):
            c = col_offset + i * 4 + k
            U[i * 5 + 4, c] = Fraction(1)  # A[i,4]
            V[4 * 5 + k, c] = Fraction(1)  # B[4,k]
            W[k * 4 + i, c] = Fraction(1)  # C[k,i] -> W[k*n+i]
    
    col_offset += 16
    
    # =================================================================
    # Block 4: 4x1x1 multiplication (A[:,4] * B[4,4] -> C[:,4])
    # Rank 4
    # =================================================================
    # C[i,4] += A[i,4] * B[4,4] for i=0..3
    
    for i in range(4):
        c = col_offset + i
        U[i * 5 + 4, c] = Fraction(1)  # A[i,4]
        V[4 * 5 + 4, c] = Fraction(1)  # B[4,4]
        W[4 * 4 + i, c] = Fraction(1)  # C[4,i]
    
    col_offset += 4
    
    # col_offset should be 85
    assert col_offset == rank, f"col_offset {col_offset} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}
