"""4x4 matrix multiplication tensor decomposition using recursive Strassen."""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 4x4 matrix multiplication using recursive Strassen.
    
    A 4x4 matrix is viewed as a 2x2 matrix of 2x2 blocks.
    We apply Strassen's algorithm at the block level (7 block multiplications),
    and each block multiplication is a 2x2 matrix multiplication, which we
    also decompose using Strassen (7 scalar multiplications each).
    
    Total rank = 7 * 7 = 49.
    """
    n, m, p = DIMENSIONS  # all 4
    
    # Strassen decomposition for 2x2 matrix multiplication.
    # For X, Y in M_2(R), Z = X * Y:
    # m1 = (X00 + X11) * (Y00 + Y11)
    # m2 = (X10 + X11) * Y00
    # m3 = X00 * (Y01 - Y11)
    # m4 = X11 * (Y10 - Y00)
    # m5 = (X00 + X01) * Y11
    # m6 = (X10 - X00) * (Y00 + Y01)
    # m7 = (X01 - X11) * (Y10 + Y11)
    #
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6
    #
    # We'll define the Strassen factors as lists of (coeff, row, col) for each matrix.
    
    # Strassen U factors: for each of the 7 multiplications, list of (coeff, i, j) for X
    strassen_U = [
        [(1, 0, 0), (1, 1, 1)],       # m1: X00 + X11
        [(1, 1, 0), (1, 1, 1)],       # m2: X10 + X11
        [(1, 0, 0)],                   # m3: X00
        [(1, 1, 1)],                   # m4: X11
        [(1, 0, 0), (1, 0, 1)],       # m5: X00 + X01
        [(1, 1, 0), (-1, 0, 0)],      # m6: X10 - X00
        [(1, 0, 1), (-1, 1, 1)],      # m7: X01 - X11
    ]
    
    # Strassen V factors: for each of the 7 multiplications, list of (coeff, i, j) for Y
    strassen_V = [
        [(1, 0, 0), (1, 1, 1)],       # m1: Y00 + Y11
        [(1, 0, 0)],                   # m2: Y00
        [(1, 0, 1), (-1, 1, 1)],      # m3: Y01 - Y11
        [(1, 1, 0), (-1, 0, 0)],      # m4: Y10 - Y00
        [(1, 1, 1)],                   # m5: Y11
        [(1, 0, 0), (1, 0, 1)],       # m6: Y00 + Y01
        [(1, 1, 0), (1, 1, 1)],       # m7: Y10 + Y11
    ]
    
    # Strassen W factors: for each of the 7 multiplications, list of (coeff, i, j) for Z
    strassen_W = [
        [(1, 0, 0), (1, 1, 1)],       # m1: Z00 += 1, Z11 += 1
        [(1, 1, 0), (-1, 1, 1)],      # m2: Z10 += 1, Z11 += -1
        [(1, 0, 1), (1, 1, 1)],       # m3: Z01 += 1, Z11 += 1
        [(1, 0, 0), (1, 1, 0)],       # m4: Z00 += 1, Z10 += 1
        [(-1, 0, 0), (1, 0, 1)],      # m5: Z00 += -1, Z01 += 1
        [(1, 1, 1)],                   # m6: Z11 += 1
        [(1, 0, 0)],                   # m7: Z00 += 1
    ]
    
    # For the outer Strassen (block level), we have 7 block multiplications.
    # Each block multiplication computes a 2x2 block product.
    # The outer Strassen formulas (for 2x2 blocks of 2x2 matrices):
    # Let A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A * B where each element is a 2x2 block.
    #
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Outer Strassen U factors: for each of the 7 block multiplications,
    # list of (coeff, block_row, block_col) for A
    outer_strassen_U = [
        [(1, 0, 0), (1, 1, 1)],       # M1: A00 + A11
        [(1, 1, 0), (1, 1, 1)],       # M2: A10 + A11
        [(1, 0, 0)],                   # M3: A00
        [(1, 1, 1)],                   # M4: A11
        [(1, 0, 0), (1, 0, 1)],       # M5: A00 + A01
        [(1, 1, 0), (-1, 0, 0)],      # M6: A10 - A00
        [(1, 0, 1), (-1, 1, 1)],      # M7: A01 - A11
    ]
    
    # Outer Strassen V factors: for each of the 7 block multiplications,
    # list of (coeff, block_row, block_col) for B
    outer_strassen_V = [
        [(1, 0, 0), (1, 1, 1)],       # M1: B00 + B11
        [(1, 0, 0)],                   # M2: B00
        [(1, 0, 1), (-1, 1, 1)],      # M3: B01 - B11
        [(1, 1, 0), (-1, 0, 0)],      # M4: B10 - B00
        [(1, 1, 1)],                   # M5: B11
        [(1, 0, 0), (1, 0, 1)],       # M6: B00 + B01
        [(1, 1, 0), (1, 1, 1)],       # M7: B10 + B11
    ]
    
    # Outer Strassen W factors: for each of the 7 block multiplications,
    # list of (coeff, block_row, block_col) for C
    outer_strassen_W = [
        [(1, 0, 0), (1, 1, 1)],       # M1: C00 += 1, C11 += 1
        [(1, 1, 0), (-1, 1, 1)],      # M2: C10 += 1, C11 += -1
        [(1, 0, 1), (1, 1, 1)],       # M3: C01 += 1, C11 += 1
        [(1, 0, 0), (1, 1, 0)],       # M4: C00 += 1, C10 += 1
        [(-1, 0, 0), (1, 0, 1)],      # M5: C00 += -1, C01 += 1
        [(1, 1, 1)],                   # M6: C11 += 1
        [(1, 0, 0)],                   # M7: C00 += 1
    ]
    
    # Use rank-47 decomposition: merge two pairs of inner Strassen terms
    # The key identity: for 2x2 matrices, we can combine m1 and m4 contributions
    # from the inner Strassen when they appear in complementary outer terms.
    #
    # Specifically, in the recursive construction, terms (outer_s, inner_s) where
    # inner_s is 0 (m1) or 3 (m4) can be paired across outer terms to save 2 multiplications.
    #
    # Strategy: For outer terms M1 and M4 (indices 0 and 3), merge inner terms 0 and 3.
    # This gives 49 - 2 = 47 total multiplications.
    
    # Build the list of (outer_s, inner_s) pairs to use, excluding the merged ones
    # and adding the merged combined terms.
    
    # Standard pairs would be all (o, i) for o in 0..6, i in 0..6
    # We merge: (0, 0) + (0, 3) -> combined term A, and (3, 0) + (3, 3) -> combined term B
    # Wait, that's not quite right. Let me think more carefully.
    
    # Actually, the rank-47 result comes from a different construction. Let me use
    # the known explicit rank-47 decomposition for 4x4 matrix multiplication.
    # 
    # The construction uses the fact that M_2(R) is a division algebra and applies
    # a modified Strassen at the block level with 7 block multiplications, but then
    # uses a 7-term decomposition for 2x2 that is optimized to share structure.
    #
    # A cleaner approach: use the 47-term decomposition from the literature.
    # The rank of 4x4 matrix multiplication over R is known to be <= 47.
    # 
    # One explicit construction: take the tensor square of Strassen-2x2 (rank 7)
    # which gives rank 49, then apply two "cancellations" using the identity:
    # For any 2x2 matrices X, Y: 
    #   (X00+X11)(Y00+Y11) = X00*Y00 + X00*Y11 + X11*Y00 + X11*Y11
    # This doesn't directly help. 
    #
    # Let me instead use a verified rank-47 construction. The known result is that
    # the border rank is 47 but the actual rank over R for 4x4 is 49 (Strassen's bound).
    # Wait - actually, the rank of 4x4 matrix multiplication over R is exactly 49?
    # No - it's known that r(M_{4,4,4}) <= 47 over C, but over R it might be different.
    #
    # Given the uncertainty, let me keep the safe rank-49 construction but add a
    # deadline guard as required by the directive.
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    import time
    start_time = time.time()
    
    col = 0
    for outer_s in range(7):  # Outer Strassen term
        for inner_s in range(7):  # Inner Strassen term
            # Deadline guard: if we're running out of time, break early
            if time.time() - start_time > 100:
                break
            # Process U: combine outer and inner U factors
            for (outer_coeff, outer_br, outer_bc) in outer_strassen_U[outer_s]:
                for (inner_coeff, inner_lr, inner_lc) in strassen_U[inner_s]:
                    # Global row = 2*outer_br + inner_lr, col = 2*outer_bc + inner_lc
                    gr = 2 * outer_br + inner_lr
                    gc = 2 * outer_bc + inner_lc
                    U[gr * m + gc][col] += outer_coeff * inner_coeff
            
            # Process V: combine outer and inner V factors
            for (outer_coeff, outer_br, outer_bc) in outer_strassen_V[outer_s]:
                for (inner_coeff, inner_lr, inner_lc) in strassen_V[inner_s]:
                    # Global row = 2*outer_br + inner_lr, col = 2*outer_bc + inner_lc
                    gr = 2 * outer_br + inner_lr
                    gc = 2 * outer_bc + inner_lc
                    V[gr * p + gc][col] += outer_coeff * inner_coeff
            
            # Process W: combine outer and inner W factors
            for (outer_coeff, outer_br, outer_bc) in outer_strassen_W[outer_s]:
                for (inner_coeff, inner_lr, inner_lc) in strassen_W[inner_s]:
                    # Global row = 2*outer_br + inner_lr, col = 2*outer_bc + inner_lc
                    gr = 2 * outer_br + inner_lr
                    gc = 2 * outer_bc + inner_lc
                    # W[k*n+i] where k=col, i=row
                    W[gc * n + gr][col] += outer_coeff * inner_coeff
            
            col += 1
        if time.time() - start_time > 100:
            break
    
    # If we didn't fill all columns due to timeout, truncate
    if col < rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
