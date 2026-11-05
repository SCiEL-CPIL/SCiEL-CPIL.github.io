import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # We will use 2 Strassen 4x4x4 and 1 Schoolbook 4x4x1
    # Rank = 49 + 49 + 16 = 114
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    # u: list of (idx, val) for A (size n*m)
    # v: list of (idx, val) for B (size m*p)
    # w: list of (idx, val) for C (size p*n)
    def add_term(u, v, w):
        nonlocal col
        for idx, val in u:
            U[idx][col] += val
        for idx, val in v:
            V[idx][col] += val
        for idx, val in w:
            W[idx][col] += val
        col += 1

    # Strassen 2x2x2 terms
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*n+i)
    # C_{00} is k=0, i=0 -> 0
    # C_{01} is k=1, i=0 -> 2
    # C_{10} is k=0, i=1 -> 1
    # C_{11} is k=1, i=1 -> 3
    
    def strassen_2x2(offset_A, offset_B, offset_C):
        # offset_A: base index for A block (0, 2, 4, 6 for 4x4)
        # offset_B: base index for B block
        # offset_C: base index for C block
        # A block is 2x2, so indices are offset_A + [0,1,2,3]
        # B block is 2x2, so indices are offset_B + [0,1,2,3]
        # C block is 2x2, so indices are offset_C + [0,1,2,3]
        
        # M1 = A00 (B00 + B11)
        u = [(offset_A + 0, 1)]
        v = [(offset_B + 0, 1), (offset_B + 3, 1)]
        w = [(offset_C + 0, 1), (offset_C + 3, -1)] # C00 + C11? No, C is the result.
        # Wait, the term is M1. M1 contributes to C00, C01, C10, C11.
        # C00 += M1, C01 += 0, C10 += 0, C11 += -M1?
        # Strassen:
        # C00 = M1 + M4 - M2 + M5
        # C01 = M3 + M7
        # C10 = M6 + M2
        # C11 = M1 - M5 + M7 + M4
        
        # So for M1:
        # w = [(offset_C + 0, 1), (offset_C + 3, 1)] # C00 and C11 get +1
        add_term(u, v, [(offset_C + 0, 1), (offset_C + 3, 1)])
        
        # M2 = (A00 + A11) B11
        u = [(offset_A + 0, 1), (offset_A + 3, 1)]
        v = [(offset_B + 3, 1)]
        w = [(offset_C + 0, -1), (offset_C + 2, 1)] # C00 -1, C10 +1
        add_term(u, v, w)
        
        # M3 = (A00 + A01) B10
        u = [(offset_A + 0, 1), (offset_A + 1, 1)]
        v = [(offset_B + 2, 1)]
        w = [(offset_C + 1, 1)] # C01 +1
        add_term(u, v, w)
        
        # M4 = A11 (B00 + B01)
        u = [(offset_A + 3, 1)]
        v = [(offset_B + 0, 1), (offset_B + 1, 1)]
        w = [(offset_C + 0, 1), (offset_C + 3, 1)] # C00 +1, C11 +1
        add_term(u, v, w)
        
        # M5 = (A10 + A11) (B00 + B01)
        u = [(offset_A + 2, 1), (offset_A + 3, 1)]
        v = [(offset_B + 0, 1), (offset_B + 1, 1)]
        w = [(offset_C + 0, 1), (offset_C + 3, -1)] # C00 +1, C11 -1
        add_term(u, v, w)
        
        # M6 = (A00 - A10) (B10 + B11)
        u = [(offset_A + 0, 1), (offset_A + 2, -1)]
        v = [(offset_B + 2, 1), (offset_B + 3, 1)]
        w = [(offset_C + 2, 1)] # C10 +1
        add_term(u, v, w)
        
        # M7 = (A01 - A11) (B00 + B01)
        u = [(offset_A + 1, 1), (offset_A + 3, -1)]
        v = [(offset_B + 0, 1), (offset_B + 1, 1)]
        w = [(offset_C + 1, 1), (offset_C + 3, 1)] # C01 +1, C11 +1
        add_term(u, v, w)
        
    # Strassen 4x4x4
    def strassen_4x4(offset_A, offset_B, offset_C):
        # A is 4x4. Blocks:
        # A00 (0,0), A01 (0,2), A10 (2,0), A11 (2,2)
        # B is 4x4. Blocks:
        # B00 (0,0), B01 (0,2), B10 (2,0), B11 (2,2)
        # C is 4x4. Blocks:
        # C00 (0,0), C01 (0,2), C10 (2,0), C11 (2,2)
        
        # M1 = A00 (B00 + B11)
        # This is a 2x2x2 multiplication of the blocks.
        # We need to call strassen_2x2 for each of the 7 products.
        # But the "A" in M1 is the block A00, which is 2x2.
        # The "B" in M1 is the block (B00 + B11), which is 2x2.
        # So we need to generate the 7 terms for the 2x2x2 multiplication of A00 and (B00+B11).
        # However, (B00+B11) is a sum of two blocks.
        # This means the V vector for the terms will have contributions from B00 and B11.
        
        # Instead of recursively calling strassen_2x2 which assumes specific offsets,
        # we can just generate the 7 terms for the 2x2x2 core, and map the indices.
        
        # Let's define a helper to generate the 7 terms for a 2x2x2 multiplication
        # given the base offsets for A, B, C.
        # But here B is a sum.
        # So for each term in the 2x2x2 decomposition, the V part will be doubled?
        # No, if B = B00 + B11, then v = v_B00 + v_B11.
        
        # Let's just write out the 49 terms explicitly using the 2x2x2 logic.
        
        # M1 = A00 (B00 + B11)
        # A00 is at offset_A + 0.
        # B00 is at offset_B + 0.
        # B11 is at offset_B + 6. (2*2 + 2)
        
        # We need the 7 terms of Strassen for A00 and (B00+B11).
        # Let's call a function that generates the 7 terms for 2x2x2
        # where A is at offA, B is at offB1 + offB2, C is at offC.
        
        def strassen_2x2_sum_B(offA, offB1, offB2, offC):
            # M1 = A00 (B00+B11) -> A00 (B00+B11)
            u = [(offA + 0, 1)]
            v = [(offB1 + 0, 1), (offB2 + 0, 1), (offB1 + 3, 1), (offB2 + 3, 1)]
            w = [(offC + 0, 1), (offC + 3, 1)]
            add_term(u, v, w)
            
            # M2 = (A00+A11) B11 -> (A00+A11) (B00+B11)_11?
            # Wait, Strassen M2 = (A00+A11) B11.
            # Here B11 is the bottom-right block of the 2x2 B matrix.
            # In our case, the 2x2 B matrix is (B00+B11).
            # So its bottom-right block is B00_11 + B11_11.
            # B00_11 is at offB1 + 3.
            # B11_11 is at offB2 + 3.
            u = [(offA + 0, 1), (offA + 3, 1)]
            v = [(offB1 + 3, 1), (offB2 + 3, 1)]
            w = [(offC + 0, -1), (offC + 2, 1)]
            add_term(u, v, w)
            
            # M3 = (A00+A01) B10
            # B10 is bottom-left of (B00+B11).
            # B00_10 is at offB1 + 2.
            # B11_10 is at offB2 + 2.
            u = [(offA + 0, 1), (offA + 1, 1)]
            v = [(offB1 + 2, 1), (offB2 + 2, 1)]
            w = [(offC + 1, 1)]
            add_term(u, v, w)
            
            # M4 = A11 (B00+B01)
            # B00+B01 is top row of (B00+B11).
            # B00_00, B00_01 at offB1+0, offB1+1.
            # B11_00, B11_01 at offB2+0, offB2+1.
            u = [(offA + 3, 1)]
            v = [(offB1 + 0, 1), (offB1 + 1, 1), (offB2 + 0, 1), (offB2 + 1, 1)]
            w = [(offC + 0, 1), (offC + 3, 1)]
            add_term(u, v, w)
            
            # M5 = (A10+A11) (B00+B01)
            u = [(offA + 2, 1), (offA + 3, 1)]
            v = [(offB1 + 0, 1), (offB1 + 1, 1), (offB2 + 0, 1), (offB2 + 1, 1)]
            w = [(offC + 0, 1), (offC + 3, -1)]
            add_term(u, v, w)
            
            # M6 = (A00-A10) (B10+B11)
            # B10+B11 is bottom row of (B00+B11).
            # B00_10, B00_11 at offB1+2, offB1+3.
            # B11_10, B11_11 at offB2+2, offB2+3.
            u = [(offA + 0, 1), (offA + 2, -1)]
            v = [(offB1 + 2, 1), (offB1 + 3, 1), (offB2 + 2, 1), (offB2 + 3, 1)]
            w = [(offC + 2, 1)]
            add_term(u, v, w)
            
            # M7 = (A01-A11) (B00+B01)
            u = [(offA + 1, 1), (offA + 3, -1)]
            v = [(offB1 + 0, 1), (offB1 + 1, 1), (offB2 + 0, 1), (offB2 + 1, 1)]
            w = [(offC + 1, 1), (offC + 3, 1)]
            add_term(u, v, w)

        # Now apply to the 4x4x4 Strassen steps
        # M1 = A00 (B00 + B11)
        strassen_2x2_sum_B(offset_A + 0, offset_B + 0, offset_B + 6, offset_C + 0)
        
        # M2 = (A00 + A11) B11
        # A00+A11 is sum of A00 and A11.
        # B11 is just B11.
        def strassen_2x2_sum_A(offA1, offA2, offB, offC):
            # M1 = (A00+A11)00 B00 -> (A00+A11)00 B00
            u = [(offA1 + 0, 1), (offA2 + 0, 1)]
            v = [(offB + 0, 1)]
            w = [(offC + 0, 1), (offC + 3, 1)]
            add_term(u, v, w)
            
            # M2 = ((A00+A11)00 + (A00+A11)11) B11
            u = [(offA1 + 0, 1), (offA2 + 0, 1), (offA1 + 3, 1), (offA2 + 3, 1)]
            v = [(offB + 3, 1)]
            w = [(offC + 0, -1), (offC + 2, 1)]
            add_term(u, v, w)
            
            # M3 = ((A00+A11)00 + (A00+A11)01) B10
            u = [(offA1 + 0, 1), (offA2 + 0, 1), (offA1 + 1, 1), (offA2 + 1, 1)]
            v = [(offB + 2, 1)]
            w = [(offC + 1, 1)]
            add_term(u, v, w)
            
            # M4 = (A00+A11)11 (B00+B01)
            u = [(offA1 + 3, 1), (offA2 + 3, 1)]
            v = [(offB + 0, 1), (offB + 1, 1)]
            w = [(offC + 0, 1), (offC + 3, 1)]
            add_term(u, v, w)
            
            # M5 = ((A00+A11)10 + (A00+A11)11) (B00+B01)
            u = [(offA1 + 2, 1), (offA2 + 2, 1), (offA1 + 3, 1), (offA2 + 3, 1)]
            v = [(offB + 0, 1), (offB + 1, 1)]
            w = [(offC + 0, 1), (offC + 3, -1)]
            add_term(u, v, w)
            
            # M6 = ((A00+A11)00 - (A00+A11)10) (B10+B11)
            u = [(offA1 + 0, 1), (offA2 + 0, 1), (offA1 + 2, -1), (offA2 + 2, -1)]
            v = [(offB + 2, 1), (offB + 3, 1)]
            w = [(offC + 2, 1)]
            add_term(u, v, w)
            
            # M7 = ((A00+A11)01 - (A00+A11)11) (B00+B01)
            u = [(offA1 + 1, 1), (offA2 + 1, 1), (offA1 + 3, -1), (offA2 + 3, -1)]
            v = [(offB + 0, 1), (offB + 1, 1)]
            w = [(offC + 1, 1), (offC + 3, 1)]
            add_term(u, v, w)

        strassen_2x2_sum_A(offset_A + 0, offset_A + 6, offset_B + 6, offset_C + 2)
        
        # M3 = (A00 + A01) B10
        # A00+A01 is sum of A00 and A01.
        # B10 is just B10.
        strassen_2x2_sum_A(offset_A + 0, offset_A + 2, offset_B + 4, offset_C + 1)
        
        # M4 = A11 (B00 + B01)
        # A11 is just A11.
        # B00+B01 is sum of B00 and B01.
        strassen_2x2_sum_B(offset_A + 6, offset_B + 0, offset_B + 2, offset_C + 3)
        
        # M5 = (A10 + A11) (B00 + B01)
        # A10+A11 sum. B00+B01 sum.
        def strassen_2x2_sum_A_B(offA1, offA2, offB1, offB2, offC):
             # M1 = (A00+A11)00 (B00+B11)00
             u = [(offA1 + 0, 1), (offA2 + 0, 1)]
             v = [(offB1 + 0, 1), (offB2 + 0, 1)]
             w = [(offC + 0, 1), (offC + 3, 1)]
             add_term(u, v, w)
             
             # M2 = ((A00+A11)00 + (A00+A11)11) (B00+B11)11
             u = [(offA1 + 0, 1), (offA2 + 0, 1), (offA1 + 3, 1), (offA2 + 3, 1)]
             v = [(offB1 + 3, 1), (offB2 + 3, 1)]
             w = [(offC + 0, -1), (offC + 2, 1)]
             add_term(u, v, w)
             
             # M3 = ((A00+A11)00 + (A00+A11)01) (B00+B11)10
             u = [(offA1 + 0, 1), (offA2 + 0, 1), (offA1 + 1, 1), (offA2 + 1, 1)]
             v = [(offB1 + 2, 1), (offB2 + 2, 1)]
             w = [(offC + 1, 1)]
             add_term(u, v, w)
             
             # M4 = (A00+A11)11 (B00+B11)00 + (B00+B11)01
             u = [(offA1 + 3, 1), (offA2 + 3, 1)]
             v = [(offB1 + 0, 1), (offB1 + 1, 1), (offB2 + 0, 1), (offB2 + 1, 1)]
             w = [(offC + 0, 1), (offC + 3, 1)]
             add_term(u, v, w)
             
             # M5 = ((A00+A11)10 + (A00+A11)11) (B00+B11)00 + (B00+B11)01
             u = [(offA1 + 2, 1), (offA2 + 2, 1), (offA1 + 3, 1), (offA2 + 3, 1)]
             v = [(offB1 + 0, 1), (offB1 + 1, 1), (offB2 + 0, 1), (offB2 + 1, 1)]
             w = [(offC + 0, 1), (offC + 3, -1)]
             add_term(u, v, w)
             
             # M6 = ((A00+A11)00 - (A00+A11)10) (B00+B11)10 + (B00+B11)11
             u = [(offA1 + 0, 1), (offA2 + 0, 1), (offA1 + 2, -1), (offA2 + 2, -1)]
             v = [(offB1 + 2, 1), (offB1 + 3, 1), (offB2 + 2, 1), (offB2 + 3, 1)]
             w = [(offC + 2, 1)]
             add_term(u, v, w)
             
             # M7 = ((A00+A11)01 - (A00+A11)11) (B00+B11)00 + (B00+B11)01
             u = [(offA1 + 1, 1), (offA2 + 1, 1), (offA1 + 3, -1), (offA2 + 3, -1)]
             v = [(offB1 + 0, 1), (offB1 + 1, 1), (offB2 + 0, 1), (offB2 + 1, 1)]
             w = [(offC + 1, 1), (offC + 3, 1)]
             add_term(u, v, w)
             
        strassen_2x2_sum_A_B(offset_A + 4, offset_A + 6, offset_B + 0, offset_B + 2, offset_C + 0)
        
        # M6 = (A00 - A10) (B10 + B11)
        # A00-A10 diff. B10+B11 sum.
        def strassen_2x2_diff_A_sum_B(offA1, offA2, offB1, offB2, offC):
             # M1 = (A00-A10)00 (B10+B11)00
             u = [(offA1 + 0, 1), (offA2 + 0, -1)]
             v = [(offB1 + 0, 1), (offB2 + 0, 1)]
             w = [(offC + 0, 1), (offC + 3, 1)]
             add_term(u, v, w)
             
             # M2 = ((A00-A10)00 + (A00-A10)11) (B10+B11)11
             u = [(offA1 + 0, 1), (offA2 + 0, -1), (offA1 + 3, 1), (offA2 + 3, -1)]
             v = [(offB1 + 3, 1), (offB2 + 3, 1)]
             w = [(offC + 0, -1), (offC + 2, 1)]
             add_term(u, v, w)
             
             # M3 = ((A00-A10)00 + (A00-A10)01) (B10+B11)10
             u = [(offA1 + 0, 1), (offA2 + 0, -1), (offA1 + 1, 1), (offA2 + 1, -1)]
             v = [(offB1 + 2, 1), (offB2 + 2, 1)]
             w = [(offC + 1, 1)]
             add_term(u, v, w)
             
             # M4 = (A00-A10)11 (B10+B11)00 + (B10+B11)01
             u = [(offA1 + 3, 1), (offA2 + 3, -1)]
             v = [(offB1 + 0, 1), (offB1 + 1, 1), (offB2 + 0, 1), (offB2 + 1, 1)]
             w = [(offC + 0, 1), (offC + 3, 1)]
             add_term(u, v, w)
             
             # M5 = ((A00-A10)10 + (A00-A10)11) (B10+B11)00 + (B10+B11)01
             u = [(offA1 + 2, 1), (offA2 + 2, -1), (offA1 + 3, 1), (offA2 + 3, -1)]
             v = [(offB1 + 0, 1), (offB1 + 1, 1), (offB2 + 0, 1), (offB2 + 1, 1)]
             w = [(offC + 0, 1), (offC + 3, -1)]
             add_term(u, v, w)
             
             # M6 = ((A00-A10)00 - (A00-A10)10) (B10+B11)10 + (B10+B11)11
             u = [(offA1 + 0, 1), (offA2 + 0, -1), (offA1 + 2, -1), (offA2 + 2, 1)]
             v = [(offB1 + 2, 1), (offB1 + 3, 1), (offB2 + 2, 1), (offB2 + 3, 1)]
             w = [(offC + 2, 1)]
             add_term(u, v, w)
             
             # M7 = ((A00-A10)01 - (A00-A10)11) (B10+B11)00 + (B10+B11)01
             u = [(offA1 + 1, 1), (offA2 + 1, -1), (offA1 + 3, -1), (offA2 + 3, 1)]
             v = [(offB1 + 0, 1), (offB1 + 1, 1), (offB2 + 0, 1), (offB2 + 1, 1)]
             w = [(offC + 1, 1), (offC + 3, 1)]
             add_term(u, v, w)
             
        strassen_2x2_diff_A_sum_B(offset_A + 0, offset_A + 4, offset_B + 4, offset_B + 6, offset_C + 2)
        
        # M7 = (A01 - A11) (B00 + B01)
        # A01-A11 diff. B00+B01 sum.
        strassen_2x2_diff_A_sum_B(offset_A + 2, offset_A + 6, offset_B + 0, offset_B + 2, offset_C + 1)
        
    # Call Strassen for first 4 columns
    strassen_4x4(0, 0, 0)
    
    # Call Strassen for next 4 columns
    # B offset for cols 4-7 is 4 in the p dimension.
    # V index is j*p + k.
    # For B, j is row (0-3), k is col (0-8).
    # Col 4-7 corresponds to k=4,5,6,7.
    # So offset_B for the block is 4?
    # B is 4x9.
    # Block B00 is rows 0-1, cols 0-1.
    # For the second Strassen, we use cols 4-7.
    # So the base offset for B in the 2nd Strassen is 4.
    # But we need to be careful with the block structure.
    # The 4x4 block of B is cols 4-7.
    # So offset_B = 4.
    strassen_4x4(0, 4, 4)
    
    # Schoolbook for last 1 column (col 8)
    # A is 4x4. B is 4x1 (col 8).
    # C is 4x1 (col 8).
    # C index k=8, i=0..3.
    # W index k*n + i = 8*4 + i = 32 + i.
    for i in range(4):
        for j in range(4):
            u = [(i*4 + j, 1)]
            v = [(j*9 + 8, 1)]
            w = [(8*4 + i, 1)]
            add_term(u, v, w)
            
    return {"U": U, "V": V, "W": W}
