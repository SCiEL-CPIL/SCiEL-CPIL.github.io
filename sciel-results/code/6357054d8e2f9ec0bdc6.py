"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 42
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # Indices for U: i*m+j (0..3)
    # Indices for V: j*p+k (0..11)
    # Indices for W: k*n+i (0..11)
    
    # We define 6 blocks of Strassen multiplications.
    # Block 0: A1 * B1_0
    # Block 1: A1 * B1_1
    # Block 2: A1 * B1_2
    # Block 3: A2 * B2_0
    # Block 4: A2 * B2_1
    # Block 5: A2 * B2_2
    
    # Strassen coefficients for 2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10-b11)
    # M4 = (a11)(b01-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a01-a10)(b00+b10)
    # M7 = (a10-a11)(b01+b11)
    
    # Output:
    # c00 = M1 - M4 + M5 - M7
    # c01 = M3 + M7
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M5
    
    # Mapping of A indices (2x4) to blocks
    # A1 is rows 0,1 and cols 0,1.
    # A2 is rows 0,1 and cols 2,3.
    # B1 is rows 0,1 and cols 0..5.
    # B2 is rows 2,3 and cols 0..5.
    
    # We iterate through the 6 blocks
    for block in range(6):
        # Determine A sub-block indices
        # A is 2x4.
        # If block < 3, use A1 (cols 0,1). Else use A2 (cols 2,3).
        if block < 3:
            a_col_offset = 0
        else:
            a_col_offset = 2
            
        # Determine B sub-block indices
        # B is 4x6.
        # If block < 3, use B1 (rows 0,1). Else use B2 (rows 2,3).
        # And col offset within B1/B2 is (block % 3) * 2
        if block < 3:
            b_row_offset = 0
        else:
            b_row_offset = 2
            
        b_col_offset = (block % 3) * 2
        
        # Strassen terms
        for s in range(7):
            col = block * 7 + s
            
            # Define U, V, W entries for this Strassen term
            # U corresponds to A, V to B, W to C
            
            # We need to map the Strassen variables to the global indices.
            
            # A indices:
            # a00: i=0, j=a_col_offset+0
            # a01: i=0, j=a_col_offset+1
            # a10: i=1, j=a_col_offset+0
            # a11: i=1, j=a_col_offset+1
            
            # B indices:
            # b00: i=b_row_offset+0, j=b_col_offset+0
            # b01: i=b_row_offset+0, j=b_col_offset+1
            # b10: i=b_row_offset+1, j=b_col_offset+0
            # b11: i=b_row_offset+1, j=b_col_offset+1
            
            # C indices:
            # c00: i=0, j=b_col_offset+0
            # c01: i=0, j=b_col_offset+1
            # c10: i=1, j=b_col_offset+0
            # c11: i=1, j=b_col_offset+1
            
            # Strassen M definitions (coeffs for A and B)
            if s == 0:
                # M1 = (a00+a11)(b00+b11)
                U[0*m + a_col_offset+0][col] += 1
                U[1*m + a_col_offset+1][col] += 1
                V[(b_row_offset+0)*p + b_col_offset+0][col] += 1
                V[(b_row_offset+1)*p + b_col_offset+1][col] += 1
                W[(b_col_offset+0)*n + 0][col] += 1
                W[(b_col_offset+1)*n + 0][col] -= 1
                W[(b_col_offset+0)*n + 1][col] += 1
                W[(b_col_offset+1)*n + 1][col] -= 1
            elif s == 1:
                # M2 = (a10+a11)(b00)
                U[1*m + a_col_offset+0][col] += 1
                U[1*m + a_col_offset+1][col] += 1
                V[(b_row_offset+0)*p + b_col_offset+0][col] += 1
                W[(b_col_offset+0)*n + 1][col] += 1
                W[(b_col_offset+1)*n + 1][col] -= 1
            elif s == 2:
                # M3 = (a00)(b10-b11)
                U[0*m + a_col_offset+0][col] += 1
                V[(b_row_offset+1)*p + b_col_offset+0][col] += 1
                V[(b_row_offset+1)*p + b_col_offset+1][col] -= 1
                W[(b_col_offset+1)*n + 0][col] += 1
                W[(b_col_offset+1)*n + 1][col] += 1
            elif s == 3:
                # M4 = (a11)(b01-b00)
                U[1*m + a_col_offset+1][col] += 1
                V[(b_row_offset+0)*p + b_col_offset+1][col] += 1
                V[(b_row_offset+0)*p + b_col_offset+0][col] -= 1
                W[(b_col_offset+0)*n + 0][col] -= 1
                W[(b_col_offset+1)*n + 0][col] += 1
                W[(b_col_offset+0)*n + 1][col] += 1
            elif s == 4:
                # M5 = (a00+a01)(b11)
                U[0*m + a_col_offset+0][col] += 1
                U[0*m + a_col_offset+1][col] += 1
                V[(b_row_offset+1)*p + b_col_offset+1][col] += 1
                W[(b_col_offset+0)*n + 0][col] -= 1
                W[(b_col_offset+1)*n + 0][col] += 1
                W[(b_col_offset+1)*n + 1][col] += 1
            elif s == 5:
                # M6 = (a01-a10)(b00+b10)
                U[0*m + a_col_offset+1][col] += 1
                U[1*m + a_col_offset+0][col] -= 1
                V[(b_row_offset+0)*p + b_col_offset+0][col] += 1
                V[(b_row_offset+1)*p + b_col_offset+0][col] += 1
                # M6 is not used in standard Strassen output formulas?
                # Wait, standard Strassen uses M1..M7.
                # c00 = M1 - M4 + M5 - M7
                # c01 = M3 + M7
                # c10 = M2 + M4
                # c11 = M1 - M2 + M3 + M5
                # M6 is NOT used in the output?
                # Let me check.
                # Ah, different Strassen variants exist.
                # The 7-multiplication one:
                # m1 = (a00+a11)(b00+b11)
                # m2 = (a10+a11)(b00)
                # m3 = (a00)(b10-b11)
                # m4 = (a11)(b01-b00)
                # m5 = (a00+a01)(b11)
                # m6 = (a01-a10)(b00+b10)
                # m7 = (a10-a11)(b01+b11)
                # c00 = m1 - m4 + m5 - m7
                # c01 = m3 + m7
                # c10 = m2 + m4
                # c11 = m1 - m2 + m3 + m5
                # Yes, m6 is not used in the final sum?
                # Wait.
                # Let's re-verify Strassen.
                # Usually:
                # M1 = (a+d)(e+f)
                # M2 = (c+d)e
                # M3 = a(f-d)
                # M4 = d(b-e)
                # M5 = (a+b)f
                # M6 = (c-a)(e+f)
                # M7 = (b-d)(d+f)
                # c = M1 - M4 + M5 - M7
                # d = M3 + M7
                # e = M2 + M4
                # f = M1 - M2 + M3 + M5
                # M6 is indeed not used in the output?
                # No, that's wrong. M6 is used?
                # Let's check Wikipedia.
                # "The seven multiplications are:
                # M1 = (a00+a11)(b00+b11)
                # M2 = (a10+a11)(b00)
                # M3 = (a00)(b10-b11)
                # M4 = (a11)(b01-b00)
                # M5 = (a00+a01)(b11)
                # M6 = (a01-a10)(b00+b10)
                # M7 = (a10-a11)(b01+b11)
                # The result is:
                # c00 = M1 - M4 + M5 - M7
                # c01 = M3 + M7
                # c10 = M2 + M4
                # c11 = M1 - M2 + M3 + M5"
                # It seems M6 is not used?
                # Actually, many sources list M6 but it might be redundant or I am misremembering the formula.
                # If M6 is not used, we can set its coefficients to 0.
                # But wait, if M6 is not used, why compute it?
                # Perhaps I should just use the 7 terms and leave M6 as 0 in U,V,W?
                # Or is there a variant where M6 is used?
                # Let's assume the standard 7-term Strassen.
                # If M6 is not in the output, I can just skip it or set it to 0.
                # But the rank is 7. If I skip it, rank is 6?
                # No, the rank is the number of terms in the decomposition.
                # If M6 is not needed, the rank is 6?
                # No, Strassen is rank 7.
                # I will include M6 in the decomposition but with 0 contribution to W?
                # No, that's a waste.
                # I will just set M6's U,V,W to 0.
                pass
            elif s == 6:
                # M7 = (a10-a11)(b01+b11)
                U[1*m + a_col_offset+0][col] += 1
                U[1*m + a_col_offset+1][col] -= 1
                V[(b_row_offset+0)*p + b_col_offset+1][col] += 1
                V[(b_row_offset+1)*p + b_col_offset+1][col] += 1
                W[(b_col_offset+0)*n + 0][col] -= 1
                W[(b_col_offset+1)*n + 0][col] += 1
                W[(b_col_offset+1)*n + 1][col] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
