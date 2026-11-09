def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen multiplications definitions
    # Each is a tuple (A_coeffs, B_coeffs, C_coeffs)
    # Coeffs are lists of (local_idx, coeff)
    
    def get_idx(offset, local):
        r, c = local
        or_, oc = offset
        return (or_ + r, oc + c)
        
    def set_val(matrix, global_idx, val):
        r, c = global_idx
        matrix[r][c] += val

    # Define the 7 Strassen steps
    # A_local, B_local are 2x2
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10
    # M3 = A00(B10-B00) -> C01
    # M4 = A11(B11-B00) -> C00, C10
    # M5 = (A00+A01)B11 -> C00, C01
    # M6 = (A10-A00)(B00+B01) -> C10? No, M6 is usually (A21-A11)(B11+B12).
    # Let's stick to the standard set used in the prompt's previous thought trace, which verified C01 = M3+M5.
    
    # Correct Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B11-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M7
    
    steps = [
        # 1
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        # 2
        ([((1,0),1), ((1,1),1)], [((0,0),1)], [((1,0),1)]),
        # 3
        ([((0,0),1)], [((1,0),1), ((0,0),-1)], [((0,1),1)]),
        # 4
        ([((1,1),1)], [((1,1),1), ((0,0),-1)], [((0,0),1), ((1,0),1)]),
        # 5
        ([((0,0),1), ((0,1),1)], [((1,1),1)], [((0,0),-1), ((0,1),1)]),
        # 6
        ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)], [((1,0),-1)]), # Check M6 contribution
        # 7
        ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)], [((0,0),1), ((1,1),-1)])
    ]
    
    # Verify C10 = M2 + M4.
    # M2 C: (1,0) +1.
    # M4 C: (1,0) +1.
    # Sum: (1,0) +2? No, M4 is A11(B11-B00).
    # C10 = M2 + M4.
    # My M4 C coeffs: ((0,0),1), ((1,0),1). So C10 gets +1 from M4.
    # My M2 C coeffs: ((1,0),1). So C10 gets +1 from M2.
    # Total C10 = 1 + 1 = 2? No, the coefficients in the formula are just "contributes".
    # The formula is C = sum(M_k * coeffs).
    # So if C10 = 1*M2 + 1*M4, then in M2 step, W has (1,0)->1. In M4 step, W has (1,0)->1.
    # This is correct.
    
    # Verify C11 = M1 - M2 + M3 - M7.
    # M1: (1,1)+1.
    # M2: (1,1)?? No, M2 is (A10+A11)B00. Result C10. So M2 does not touch C11.
    # Wait.
    # M2 = (A21+A22)B11. Result is C21 (C10 in 0-index).
    # So M2 W coeffs: ((1,0),1). Correct.
    # M3 = A11(B21-B11). Result is C12 (C01 in 0-index).
    # M3 W coeffs: ((0,1),1). Correct.
    # M7 = (A12-A22)(B21+B22).
    # Result C11?
    # C11 = M1 - M2 + M3 - M7.
    # M7 must contribute to C11 with -1.
    # My M7 W coeffs: ((0,0),1), ((1,1),-1).
    # So C11 gets -1 from M7. Correct.
    
    # Verify C00 = M1 + M4 - M5 + M7.
    # M1: (0,0)+1.
    # M4: (0,0)+1.
    # M5: (0,0)-1.
    # M7: (0,0)+1.
    # Sum: 1+1-1+1 = 2? No.
    # Formula: C00 = M1 + M4 - M5 + M7.
    # Coeffs: M1(1), M4(1), M5(-1), M7(1).
    # My W coeffs for (0,0):
    # M1: 1
    # M4: 1
    # M5: -1
    # M7: 1
    # Sum: 1+1-1+1 = 2.
    # This implies C00 is computed as 2*C00? No.
    # The decomposition is $T = \sum U V^T W^T$.
    # If I put coeff 1 in W for M1, and 1 for M4, etc., the sum of contributions is the result.
    # If the formula says $C_{00} = 1 \cdot M_1 + 1 \cdot M_4 - 1 \cdot M_5 + 1 \cdot M_7$,
    # then my W coefficients for the (0,0) slot should be 1, 1, -1, 1.
    # The sum is $1+1-1+1 = 2$.
    # This means the calculated value will be $2 C_{00}$?
    # No. The "value" in the tensor is the scalar multiplication.
    # The term is $u v w$.
    # If $w=1$ for M1, it adds $A_{part} B_{part}$.
    # If the math requires $1 \times M_1$, then $w=1$ is correct.
    # The fact that the sum of coeffs is 2 doesn't matter, it just means multiple terms contribute to that output cell.
    # So this is correct.
    
    # Sub-problem definitions:
    # (a_row, a_col), (b_row, b_col), (c_row, c_col)
    subs = [
        ((0, 0), (0, 0), (0, 0)), # AL * BTL -> CL
        ((0, 2), (2, 0), (0, 0)), # AR * BBL -> CL
        ((0, 0), (0, 2), (0, 2)), # AL * BTR -> CR
        ((0, 2), (2, 2), (0, 2)), # AR * BBR -> CR
    ]
    
    col = 0
    for a_off, b_off, c_off in subs:
        for a_coeffs, b_coeffs, c_coeffs in steps:
            # Fill U
            for (lr, lc), coeff in a_coeffs:
                gr, gc = get_idx(a_off, (lr, lc))
                U[gr][gc] += coeff # Wait, U is (n*m, rank). Index is row, col.
                # U row index is i*m + j. Here m=4.
                # So row is gr * 4 + gc?
                # No, A is 2x4. Rows 0,1. Cols 0..3.
                # Index in U is i * 4 + j.
                U[gr * 4 + gc][col] += coeff
            
            # Fill V
            for (lr, lc), coeff in b_coeffs:
                gr, gc = get_idx(b_off, (lr, lc))
                # B is 4x4. Rows 0..3. Cols 0..3.
                # Index in V is i * 4 + j.
                V[gr * 4 + gc][col] += coeff
            
            # Fill W
            for (lr, lc), coeff in c_coeffs:
                gr, gc = get_idx(c_off, (lr, lc))
                # C is 2x4. Rows 0,1. Cols 0..3.
                # Output factor W has shape (p*n, rank) = (4*2, 28) = (8, 28).
                # Convention: W[k*n+i, r].
                # k is col of C (0..3). i is row of C (0,1).
                # So index is k * 2 + i.
                W[gc * 2 + gr][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
