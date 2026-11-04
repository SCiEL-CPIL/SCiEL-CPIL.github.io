def add_strassen_block(U, V, W, rank_offset, i_base, j_base, k_base):
    # i_base, j_base, k_base are offsets for the 2x2x2 block
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # Strassen multiplications M1..M7
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Note: A is i x j. B is j x k.
    # Indices in A: (i, j). Indices in B: (j, k).
    # U[i, j, r], V[j, k, r], W[k, i, r]
    
    terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # Each coeff is a list of (index, value)
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # M2
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((0,1), 1)]), # C10 += M2? No C10 = M2+M4. C[1,0] is i=1, k=0. W[0,1,r]
        # M3
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((1,0), 1)]), # C01 += M3. C[0,1] is i=0, k=1. W[1,0,r]
        # M4
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,1), 1), ((1,0), 1)]), # C00 += M4, C10 += M4
        # M5
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((1,0), -1), ((1,1), 1)]), # C00 -= M5, C11 += M5? No C00 = ... -M5. C01 = M3+M5.
        # Correction on Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's re-verify M5 contribution.
        # M5 = (A00+A01)B11.
        # C01 += M5. C[0,1] -> W[1,0,r] += 1.
        # C00 -= M5. C[0,0] -> W[0,0,r] -= 1.
        
        # M6
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]), # C11 += M6.
        # M7
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]), # C00 += M7.
    ]
    
    # I need to be very careful with the W indices.
    # W[k, i, r].
    # C[i, k] gets contribution.
    # So W[k*n + i, r] += coeff.
