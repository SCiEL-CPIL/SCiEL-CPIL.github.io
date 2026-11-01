"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 using 4 independent Strassen decompositions
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 terms for 2x2x2
    # u, v, c matrices (2x2 each)
    terms = [
        # M1
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        # M2
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, -1]]),
        # M3
        ([[1, 0], [0, 0]], [[0, 0], [1, -1]], [[0, 1], [0, 1]]),
        # M4
        ([[0, 0], [0, 1]], [[-1, 1], [0, 0]], [[1, 0], [1, 0]]),
        # M5
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]),
        # M6
        ([-1, 0], [1, 1], [0, 0]), # Placeholder, need correct M6
        # M7
        ([[0, 1], [-1, 0]], [[0, 1], [1, 1]], [[0, -1], [1, 0]]) # Placeholder
    ]
    
    # Correct Strassen Coefficients
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Let's define the 7 (u, v, w) triples explicitly.
    # u is 2x2, v is 2x2, w is 2x2 (output coefficients)
    
    strassen_terms = [
        # 1
        (
            [[1, 0], [0, 1]],  # u
            [[1, 0], [0, 1]],  # v
            [[1, 0], [0, 1]]   # w (C00+=1, C11+=1)
        ),
        # 2
        (
            [[0, 0], [1, 1]],  # u
            [[1, 0], [0, 0]],  # v
            [[0, 0], [1, -1]]  # w (C10+=1, C11-=1)
        ),
        # 3
        (
            [[1, 0], [0, 0]],  # u
            [[0, 0], [1, -1]], # v
            [[0, 1], [0, 1]]   # w (C01+=1, C11+=1)
        ),
        # 4
        (
            [[0, 0], [0, 1]],  # u
            [[-1, 1], [0, 0]], # v
            [[1, 0], [1, 0]]   # w (C00+=1, C10+=1)
        ),
        # 5
        (
            [[1, 1], [0, 0]],  # u
            [[0, 0], [0, 1]],  # v
            [[-1, 1], [0, 0]]  # w (C00-=1, C01+=1)
        ),
        # 6
        (
            [[-1, 0], [1, 0]], # u
            [[1, 1], [0, 0]],  # v
            [[0, 0], [0, 0]]   # w (M6 is not in output? Wait.)
        ),
        # 7
        (
            [[0, 1], [-1, 0]], # u
            [[0, 1], [1, 1]],  # v
            [[0, -1], [1, 0]]  # w (C00-=1, C11+=1)
        )
    ]
    
    # Verify M6.
    # Standard Strassen:
    # M6 = (A10 - A00)(B00 + B01)
    # This term is used to cancel out unwanted terms in other combinations?
    # Actually, the standard 7 mults are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Outputs:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Note that M6 is NOT in the output formulas!
    # Is this correct?
    # Let's check the number of multiplications. 7.
    # If M6 is not used, why compute it?
    # Ah, I might be misremembering the standard set.
    # Let's check a reliable source or derive it.
    # Actually, there are many equivalent Strassen decompositions.
    # One common set uses M6 in the outputs?
    # No, usually 7 terms are used.
    # Let's check if the above 7 terms (excluding M6 from outputs) sum to the correct C.
    
    # Let's re-evaluate C00 with the terms above.
    # C00 = 1*M1 + 1*M4 - 1*M5 - 1*M7
    # We already checked this and it didn't work out.
    
    # Let's use a known valid set of 7 tensors for 2x2x2.
    # From literature:
    # T1: u=(1,0;0,1), v=(1,0;0,1), w=(1,0;0,1)
    # T2: u=(0,0;1,1), v=(1,0;0,0), w=(0,0;1,-1)
    # T3: u=(1,0;0,0), v=(0,0;1,-1), w=(0,1;0,1)
    # T4: u=(0,0;0,1), v=(-1,1;0,0), w=(1,0;1,0)
    # T5: u=(1,1;0,0), v=(0,0;0,1), w=(-1,1;0,0)
    # T6: u=(0,1;-1,0), v=(0,1;1,1), w=(0,-1;1,0)
    # T7: u=(-1,0;1,0), v=(1,1;0,0), w=(0,0;0,0) ??
    
    # Let's just use the schoolbook rank 8 for 2x2x2?
    # Rank 8 * 4 = 32.
    # Current is 64.
    # 32 is better than 64.
    # 28 is better than 32.
    # I should implement Strassen correctly.
    
    # Correct Strassen Decomposition (from Wikipedia or similar):
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)b00
    # m3 = a00(b10 - b11)
    # m4 = a11(b01 - b00)
    # m5 = (a00 + a01)b11
    # m6 = (a10 - a00)(b00 + b01)
    # m7 = (a01 - a11)(b10 + b11)
    
    # c00 = m1 + m4 - m5 - m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m7
    
    # Let's re-verify c00.
    # m1 = a00b00 + a00b11 + a11b00 + a11b11
    # m4 = a11b01 - a11b00
    # m5 = a00b11 + a01b11
    # m7 = a01b10 + a01b11 - a11b10 - a11b11
    
    # c00 = (a00b00 + a00b11 + a11b00 + a11b11) + (a11b01 - a11b00) - (a00b11 + a01b11) - (a01b10 + a01b11 - a11b10 - a11b11)
    # = a00b00 + a11b01 - a01b11 - a01b10 + a11b10 + a11b11
    # This is NOT c00 = a00b00 + a01b10.
    
    # There is a mistake in my manual expansion or the formula.
    # Actually, c00 = a00b00 + a01b10.
    # My expansion has extra terms.
    
    # Let's try a different set of Strassen terms.
    # Or just use Rank 8 (Schoolbook) for 2x2x2.
    # Rank 8 is easy to implement correctly.
    # 4 * 8 = 32.
    # 32 < 64.
    # This is a valid improvement.
    
    # Let's implement Rank 8 for each block.
    
    for l in range(4):
        base_col = l * 8
        # For each i, j, k' in 0..1
        for i in range(2):
            for j in range(2):
                for kp in range(2):
                    col = base_col + i * 4 + j * 2 + kp
                    # U index: i*2 + j
                    U[i * 2 + j][col] = 1
                    # V index: j*16 + (2l + kp)
                    V[j * 16 + (2 * l + kp)][col] = 1
                    # W index: (2l + kp)*2 + i
                    W[(2 * l + kp) * 2 + i][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
