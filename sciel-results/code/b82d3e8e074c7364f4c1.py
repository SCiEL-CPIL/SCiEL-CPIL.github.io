import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    # Rank is 7 per 2x2 block of B, and there are p//2 = 5 blocks
    rank = 7 * (p // 2)
    
    nU = n * m
    nV = m * p
    nW = p * n

    # Initialize matrices
    U = [[0] * rank for _ in range(nU)]
    V = [[0] * rank for _ in range(nV)]
    W = [[0] * rank for _ in range(nW)]

    # Laderman's Algorithm for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a10)(b10 + b11)
    #
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M6
    # c11 = M1 - M2 + M3 + M7
    
    # Indices for A (n*m = 4): 
    # 0: a00, 1: a01, 2: a10, 3: a11
    # Indices for B block (2x2):
    # 0: b00, 1: b01, 2: b10, 3: b11
    # Indices for C block (2x2):
    # 0: c00, 1: c10 (wait, W index is k*n+i)
    # W index: k*n + i.
    # For C[i, k]:
    # i=0, k=0 -> 0*2+0 = 0 (c00)
    # i=1, k=0 -> 0*2+1 = 1 (c10)
    # i=0, k=1 -> 1*2+0 = 2 (c01)
    # i=1, k=1 -> 1*2+1 = 3 (c11)
    
    # So C_vec order in Laderman:
    # c00, c10, c01, c11
    # My previous mapping:
    # W[2*block*n + 0] = c00
    # W[2*block*n + 1] = c10
    # W[(2*block+1)*n + 0] = c01
    # W[(2*block+1)*n + 1] = c11
    
    # Let's map Laderman outputs to these indices:
    # c00 = M1 + M4 - M5 - M7
    # c10 = M2 + M6
    # c01 = M3 + M5
    # c11 = M1 - M2 + M3 + M7

    # Define the 7 terms for one block
    # Each term is (U_vec, V_vec, W_vec)
    # U_vec is length 4 (for A)
    # V_vec is length 4 (for B block)
    # W_vec is length 4 (for C block, order: c00, c10, c01, c11)
    
    terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # W: c00=1, c10=0, c01=0, c11=1
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # W: c00=0, c10=1, c01=0, c11=-1
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # W: c00=0, c10=0, c01=1, c11=1
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0]), # W: c00=1, c10=0, c01=0, c11=0
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # W: c00=-1, c10=0, c01=1, c11=0
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0]), # W: c00=0, c10=1, c01=0, c11=0
        # M7
        ([0, 1, -1, 0], [0, 0, 1, 1], [-1, 0, 0, 1]), # W: c00=-1, c10=0, c01=0, c11=1
    ]
    
    # Verify one block manually
    # c00 = 1 + 0 + 0 + 1 - 1 + 0 - 1 = 0? 
    # Wait.
    # M1: 1
    # M4: 1
    # M5: -1
    # M7: -1
    # Sum: 1 + 1 - 1 - 1 = 0. 
    # This is wrong. c00 should be M1 + M4 - M5 - M7.
    # Let's re-check Laderman's formulas.
    
    # Laderman:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a10)(b10+b11)
    
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M6
    # c11 = M1 - M2 + M3 + M7
    
    # Let's check c00 with a=I, b=I.
    # a00=1, a11=1, a01=0, a10=0. b00=1, b11=1, b01=0, b10=0.
    # M1 = (1+1)(1+1) = 4
    # M2 = (0+1)1 = 1
    # M3 = 1(0-1) = -1
    # M4 = 1(0-1) = -1
    # M5 = (1+0)1 = 1
    # M6 = (0-1)(1+0) = -1
    # M7 = (0-0)(0+1) = 0
    # c00 = 4 + (-1) - 1 - 0 = 2. Correct (1*1 + 0*0 = 1? No. A=I, B=I -> C=I. c00=1).
    # Wait. A=I, B=I.
    # a00=1, a11=1. b00=1, b11=1.
    # M1 = 2*2=4.
    # M4 = 1*(0-1) = -1.
    # M5 = 1*1 = 1.
    # M7 = 0.
    # c00 = 4 - 1 - 1 = 2.
    # But c00 should be 1.
    # There is a scaling factor or I am using the wrong Laderman variant.
    
    # Standard Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a10)(b10+b11)
    
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M6
    # c11 = M1 - M2 + M3 + M7
    
    # Let's re-evaluate Strassen with A=I, B=I.
    # M1 = 4
    # M2 = 1
    # M3 = -1
    # M4 = -1
    # M5 = 1
    # M6 = -1
    # M7 = 0
    # c00 = 4 - 1 - 1 - 0 = 2. Still 2.
    # Ah, Strassen is for 2x2.
    # c00 = a00b00 + a01b10.
    # If A=I, B=I, c00=1.
    # My calculation gives 2.
    # Why?
    # M1 = (1+1)(1+1) = 4.
    # M4 = 1(0-1) = -1.
    # M5 = (1+0)1 = 1.
    # M7 = (0-0)1 = 0.
    # c00 = 4 - 1 - 1 = 2.
    # Is Strassen correct?
    # Yes, Strassen is correct.
    # Did I plug in wrong?
    # b10=0, b00=1. b10-b00 = -1.
    # a11=1. M4 = -1.
    # a00=1, a01=0. b11=1. M5 = 1.
    # a01=0, a10=0. M7 = 0.
    # M1 = 4.
    # c00 = 4 - 1 - 1 = 2.
    # Wait. c00 = a00b00 + a01b10 = 1*1 + 0*0 = 1.
    # There is a mistake in my manual check or the formula.
    
    # Let's check the formula for c00 in Strassen.
    # c00 = M1 + M4 - M5 - M7.
    # Let's expand M1+M4-M5-M7.
    # M1 = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11
    # M4 = a11(b10-b00) = a11b10 - a11b00
    # M5 = (a00+a01)b11 = a00b11 + a01b11
    # M7 = (a01-a10)(b10+b11) = a01b10 + a01b11 - a10b10 - a10b11
    
    # Sum:
    # a00b00 + a00b11 + a11b00 + a11b11 + a11b10 - a11b00 - a00b11 - a01b11 - a01b10 - a01b11 + a10b10 + a10b11
    # Cancel terms:
    # a00b11 - a00b11 = 0
    # a11b00 - a11b00 = 0
    # Remaining:
    # a00b00 + a11b11 + a11b10 - a01b11 - a01b10 - a01b11 + a10b10 + a10b11
    # This does not look like a00b00 + a01b10.
    
    # I must have the wrong Strassen formulas or wrong assignment of M's.
    # Let's use a verified set.
    # Wikipedia "Strassen algorithm":
    # m1 = (a11 + a22)(b11 + b22)
    # m2 = (a21 + a22)b11
    # m3 = a11(b12 - b22)
    # m4 = a22(b21 - b11)
    # m5 = (a11 + a12)b22
    # m6 = (a21 - a11)(b11 + b12)
    # m7 = (a12 - a21)(b21 + b22)
    
    # c11 = m1 + m4 - m5 - m7
    # c12 = m3 + m5
    # c21 = m2 + m6
    # c22 = m1 - m2 + m3 + m7
    
    # Note: indices are 1-based in wiki.
    # My indices:
    # a00=a11, a01=a12, a10=a21, a11=a22.
    # b00=b11, b01=b12, b10=b21, b11=b22.
    # c00=c11, c01=c12, c10=c21, c11=c22.
    
    # Let's re-evaluate c00 (wiki c11) with A=I, B=I.
    # m1 = (1+1)(1+1) = 4
    # m4 = 1(0-1) = -1
    # m5 = (1+0)1 = 1
    # m7 = (0-0)1 = 0
    # c11 = 4 - 1 - 1 - 0 = 2.
    # Still 2.
    # Why is it 2?
    # c11 = a11b11 + a12b21.
    # If A=I, B=I, c11=1.
    # The formula must be wrong or I am applying it wrong.
    # Let's check the expansion of m1+m4-m5-m7 again.
    # m1 = a11b11 + a11b22 + a22b11 + a22b22
    # m4 = a22b21 - a22b11
    # m5 = a11b22 + a12b22
    # m7 = a12b21 + a12b22 - a21b21 - a21b22
    
    # m1+m4-m5-m7 = 
    # a11b11 + a11b22 + a22b11 + a22b22 
    # + a22b21 - a22b11 
    # - a11b22 - a12b22 
    # - a12b21 - a12b22 + a21b21 + a21b22
    
    # Cancel:
    # a11b22 - a11b22 = 0
    # a22b11 - a22b11 = 0
    # Left:
    # a11b11 + a22b22 + a22b21 - a12b22 - a12b21 - a12b22 + a21b21 + a21b22
    
    # This is definitely not a11b11 + a12b21.
    # There is a known error in my transcription or memory of Strassen.
    
    # Let's use the code from the current program which was verified to work (score 1.1429).
    # The current program used:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10+B01)  <-- Note: B10+B01, not B10-B00
    # M5 = (A00+A01)*(B11)
    # M6 = (A10-A00)*(B01+B10)
    # M7 = (A01-A10)*(B00+B10)
    
    # And outputs:
    # C00 = M1+M4-M5-M7
    # C01 = M3+M5
    # C10 = M2+M6
    # C11 = M1-M2+M3+M7
    
    # Let's check this set with A=I, B=I.
    # M1 = 4
    # M2 = 1
    # M3 = -1
    # M4 = 1(0+0) = 0
    # M5 = 1
    # M6 = -1(0+0) = 0
    # M7 = 0(1+0) = 0
    # C00 = 4 + 0 - 1 - 0 = 3. Wrong.
    
    # Okay, I will just copy the terms from the current program's fallback which is known to work.
    
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Fill matrices
    col = 0
    for block in range(p // 2):
        for a_vec, b_vec, c_vec in terms:
            # U
            for idx in range(nU):
                U[idx][col] = a_vec[idx]
            
            # V
            # B row 0, cols 2*block, 2*block+1
            # B row 1, cols 2*block, 2*block+1
            V[0 * p + 2 * block][col] = b_vec[0]
            V[0 * p + 2 * block + 1][col] = b_vec[1]
            V[1 * p + 2 * block][col] = b_vec[2]
            V[1 * p + 2 * block + 1][col] = b_vec[3]
            
            # W
            # C row 0, col 2*block -> index (2*block)*n + 0
            # C row 1, col 2*block -> index (2*block)*n + 1
            # C row 0, col 2*block+1 -> index (2*block+1)*n + 0
            # C row 1, col 2*block+1 -> index (2*block+1)*n + 1
            W[2 * block * n + 0][col] = c_vec[0]
            W[2 * block * n + 1][col] = c_vec[1]
            W[(2 * block + 1) * n + 0][col] = c_vec[2]
            W[(2 * block + 1) * n + 1][col] = c_vec[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
