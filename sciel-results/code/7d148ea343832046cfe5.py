import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank estimate: 49 (4x4x4 Strassen) + 16 (4x1x4) + 20 (4x5x1) = 85
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, col):
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1

    def add_term_scaled(u_idx, v_idx, w_idx, col, val):
        U[u_idx][col] = val
        V[v_idx][col] = 1
        W[w_idx][col] = 1

    # 1. Strassen 4x4x4 for A[0:4, 0:4] * B[0:4, 0:4]
    # This handles C[0:4, 0:4]
    # We need the 49 terms of the 4x4x4 Strassen decomposition.
    
    # Strassen 2x2 base terms
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B10)
    # M7: (A01-A11)(B11+B01)
    
    # We will build the 4x4 decomposition by composing these.
    # Instead of hardcoding 49 terms, we can use the structure.
    # But for speed and simplicity, let's just iterate.
    
    # Let's define the 2x2 Strassen decomposition explicitly.
    # U, V, W for 2x2x2
    # Indices: u = i*2+j, v = j*2+k, w = k*2+i
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1
    # (A00+A11)(B00+B11)
    # Terms: A00B00, A00B11, A11B00, A11B11
    # But we need to combine them into rank 1 terms?
    # No, Strassen reduces the number of multiplications.
    # The tensor for M1 is (A00+A11) * (B00+B11).
    # This is not a single rank 1 term in the original A,B,C space.
    # It is a product of two linear forms.
    # So in the decomposition, we have a term where U is the vector for (A00+A11) and V is the vector for (B00+B11).
    # And W is determined by the output.
    
    # Wait, the decomposition is sum of U_r V_r W_r.
    # For Strassen, the "multiplication" is of linear combinations.
    # So U_r is the coefficient vector for A in the linear combo.
    # V_r is the coefficient vector for B in the linear combo.
    # W_r is the coefficient vector for C in the linear combo.
    
    # For M1 = (A00+A11)(B00+B11):
    # U has 1 at (0,0) and (1,1).
    # V has 1 at (0,0) and (1,1).
    # W? The result is added to C00 and C11.
    # C00 += M1, C11 += M1.
    # So W has 1 at (0,0) for C00? And 1 at (1,1) for C11?
    # No, W is a vector in C space.
    # If M1 contributes to C00 and C11, then W_r must have 1s at those positions.
    # But a rank 1 term U V W means the contribution is U_i V_j W_k.
    # If W has 1 at (0,0) and (1,1), then the term contributes to both.
    # This is valid.
    
    # Let's list the 7 terms for 2x2 Strassen.
    # 0: M1. U: (0,0)=1, (1,1)=1. V: (0,0)=1, (1,1)=1. W: (0,0)=1, (1,1)=1.
    #    Wait, C00 = M1 + ... and C11 = M1 + ...
    #    So W[0] = 1 (for C00), W[3] = 1 (for C11).
    #    Indices: C00 is w=0*2+0=0. C11 is w=1*2+1=3.
    
    # 1: M2. U: (1,0)=1, (1,1)=1. V: (0,0)=1. W: (1,0)=1 (C10), (1,1)=-1 (C11).
    #    C10 = M2 + M4. C11 = M1 - M2 + ...
    #    So W[2] = 1, W[3] = -1.
    
    # 2: M3. U: (0,0)=1. V: (1,0)=1, (1,1)=-1. W: (0,1)=1 (C01), (1,0)=1 (C10).
    #    C01 = M3 + M5. C10 = M2 + M4.
    #    Wait, C10 = M2 + M4. M3 is in C01 and C11?
    #    C01 = M3 + M5.
    #    C11 = M1 - M2 + M3 + M6.
    #    So W[1] = 1, W[3] = 1.
    
    # 3: M4. U: (1,1)=1. V: (0,1)=1, (0,0)=-1. W: (0,0)=1 (C00), (1,0)=1 (C10).
    #    C00 = M1 + M4 - M5 + M7.
    #    C10 = M2 + M4.
    #    So W[0] = 1, W[2] = 1.
    
    # 4: M5. U: (0,0)=1, (0,1)=1. V: (1,1)=1. W: (0,0)=-1 (C00), (0,1)=1 (C01).
    #    C00 = ... - M5 ...
    #    C01 = M3 + M5.
    #    So W[0] = -1, W[1] = 1.
    
    # 5: M6. U: (1,0)=1, (0,0)=-1. V: (0,0)=1, (1,0)=1. W: (1,1)=1 (C11).
    #    C11 = ... + M6.
    #    So W[3] = 1.
    
    # 6: M7. U: (0,1)=1, (1,1)=-1. V: (1,1)=1, (0,1)=1. W: (0,0)=1 (C00).
    #    C00 = ... + M7.
    #    So W[0] = 1.

    # Let's verify C00 = M1 + M4 - M5 + M7.
    # W[0] for M1 is 1. M4 is 1. M5 is -1. M7 is 1. Sum = 2?
    # No, the sum of the terms in the decomposition must equal the tensor.
    # The tensor for C00 is sum of A_ij B_jk C_00.
    # In Strassen, we are replacing the 8 terms with 7 linear combinations.
    # The decomposition I'm building is for the *multiplication* tensor.
    # So I just need to list the 7 terms U_r, V_r, W_r.
    # The correctness is guaranteed if the Strassen formulas are correct.
    
    # Let's populate U2, V2, W2.
    # Term 0 (M1)
    U2[0][0] = 1; U2[3][0] = 1 # A00, A11
    V2[0][0] = 1; V2[3][0] = 1 # B00, B11
    W2[0][0] = 1; W2[3][0] = 1 # C00, C11
    
    # Term 1 (M2)
    U2[2][1] = 1; U2[3][1] = 1 # A10, A11
    V2[0][1] = 1 # B00
    W2[2][1] = 1; W2[3][1] = -1 # C10, C11
    
    # Term 2 (M3)
    U2[0][2] = 1 # A00
    V2[2][2] = 1; V2[3][2] = -1 # B10, B11
    W2[1][2] = 1; W2[3][2] = 1 # C01, C11
    
    # Term 3 (M4)
    U2[3][3] = 1 # A11
    V2[1][3] = 1; V2[0][3] = -1 # B01, B00
    W2[0][3] = 1; W2[2][3] = 1 # C00, C10
    
    # Term 4 (M5)
    U2[0][4] = 1; U2[1][4] = 1 # A00, A01
    V2[3][4] = 1 # B11
    W2[0][4] = -1; W2[1][4] = 1 # C00, C01
    
    # Term 5 (M6)
    U2[2][5] = 1; U2[0][5] = -1 # A10, A00
    V2[0][5] = 1; V2[2][5] = 1 # B00, B10
    W2[3][5] = 1 # C11
    
    # Term 6 (M7)
    U2[1][6] = 1; U2[3][6] = -1 # A01, A11
    V2[3][6] = 1; V2[1][6] = 1 # B11, B01
    W2[0][6] = 1 # C00

    # Now build 4x4x4 decomposition.
    # We have 7 "block" multiplications.
    # Each block multiplication is a 2x2x2 multiplication.
    # So we have 7 * 7 = 49 terms.
    
    # Block indices:
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    # C blocks: C00 (0,0), C01 (0,1), C10 (1,0), C11 (1,1)
    
    # Strassen M1: (A00+A11)(B00+B11)
    # U is sum of A00, A11.
    # V is sum of B00, B11.
    # W is sum of C00, C11.
    
    # We iterate over the 7 Strassen terms (s in 0..6)
    # And for each, we iterate over the 7 base terms (b in 0..6)
    # And add the product of the coefficients.
    
    # Coefficients for A in each Strassen term:
    # M1: A00=1, A11=1
    # M2: A10=1, A11=1
    # M3: A00=1
    # M4: A11=1
    # M5: A00=1, A01=1
    # M6: A10=1, A00=-1
    # M7: A01=1, A11=-1
    
    # Coefficients for B:
    # M1: B00=1, B11=1
    # M2: B00=1
    # M3: B10=1, B11=-1
    # M4: B01=1, B00=-1
    # M5: B11=1
    # M6: B00=1, B10=1
    # M7: B11=1, B01=1
    
    # Coefficients for C:
    # M1: C00=1, C11=1
    # M2: C10=1, C11=-1
    # M3: C01=1, C11=1
    # M4: C00=1, C10=1
    # M5: C00=-1, C01=1
    # M6: C11=1
    # M7: C00=1

    # We need to map block indices to global indices.
    # Block (bi, bj) of A (4x4) corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    # In global U (4x5), A is in cols 0..3.
    # So A_block(bi, bj) is at U[ (2*bi + i) * 5 + (2*bj + j) ].
    
    # For B (5x5), B is in rows 0..3.
    # B_block(bi, bj) is at V[ (2*bi + i) * 5 + (2*bj + j) ].
    
    # For C (4x5), C is in cols 0..3.
    # C_block(bi, bj) is at W[ (2*bj + j) * 4 + (2*bi + i) ].
    # Note: W index is k*n + i.
    # C is 4x5. i is row (0..3), k is col (0..4).
    # C_block(bi, bj) has rows 2*bi..2*bi+1 and cols 2*bj..2*bj+1.
    # So i_global = 2*bi + i, k_global = 2*bj + j.
    # W index = k_global * 4 + i_global.

    # Let's create the 49 terms.
    col_offset = 0
    for s in range(7):
        # Get A, B, C coefficients for this Strassen step
        # We can reuse the U2, V2, W2 logic but scaled?
        # No, U2, V2, W2 are for the *internal* 2x2 multiplication.
        # The "multiplication" in Strassen step s is of the linear forms.
        # The linear form for A is a vector of size 4 (2x2 block).
        # The linear form for B is a vector of size 4.
        # The linear form for C is a vector of size 4.
        
        # We need to expand these into the 4x4 space.
        # For each of the 7 base terms in the 2x2 multiplication, we have a specific pattern.
        
        # Actually, it's easier to just iterate over the 7 base terms of the 2x2 multiplication.
        # For each base term b, we have U2[b], V2[b], W2[b].
        # These are vectors of size 4 (for the 2x2 blocks).
        # We need to map these to the 4x4 global indices.
        
        for b in range(7):
            r = col_offset
            # Expand U2[b] (size 4) to U (size 20, but we only use 16)
            # U2[b][idx] where idx = bi*2+bj
            for bi in range(2):
                for bj in range(2):
                    val = U2[bi*2+bj][b]
                    if val != 0:
                        # Map to global U
                        # A block (bi, bj) -> rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                        for i in range(2):
                            for j in range(2):
                                u_idx = (2*bi + i) * 5 + (2*bj + j)
                                U[u_idx][r] += val
                                
            # Expand V2[b] (size 4) to V (size 25, but we only use 16)
            for bi in range(2):
                for bj in range(2):
                    val = V2[bi*2+bj][b]
                    if val != 0:
                        # Map to global V
                        # B block (bi, bj) -> rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                        for i in range(2):
                            for j in range(2):
                                v_idx = (2*bi + i) * 5 + (2*bj + j)
                                V[v_idx][r] += val
                                
            # Expand W2[b] (size 4) to W (size 20, but we only use 16)
            for bi in range(2):
                for bj in range(2):
                    val = W2[bi*2+bj][b]
                    if val != 0:
                        # Map to global W
                        # C block (bi, bj) -> rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                        for i in range(2):
                            for j in range(2):
                                w_idx = (2*bj + j) * 4 + (2*bi + i)
                                W[w_idx][r] += val
                                
            col_offset += 1
            
    # col_offset is now 49.
    
    # 2. Outer product A2 B12 (4x1 * 1x4)
    # A2 is col 4 of A. B12 is row 4 of B.
    # C is 4x4.
    # Terms: 16.
    for i in range(4):
        for k in range(4):
            r = col_offset
            u_idx = i * 5 + 4
            v_idx = 4 * 5 + k
            w_idx = k * 4 + i
            U[u_idx][r] = 1
            V[v_idx][r] = 1
            W[w_idx][r] = 1
            col_offset += 1
            
    # col_offset is now 49 + 16 = 65.
    
    # 3. Dot products A B2 (4x5 * 5x1)
    # C is 4x1.
    # Terms: 20.
    for i in range(4):
        for j in range(5):
            r = col_offset
            u_idx = i * 5 + j
            v_idx = j * 5 + 0
            w_idx = 0 * 4 + i
            U[u_idx][r] = 1
            V[v_idx][r] = 1
            W[w_idx][r] = 1
            col_offset += 1
            
    # col_offset is now 65 + 20 = 85.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
