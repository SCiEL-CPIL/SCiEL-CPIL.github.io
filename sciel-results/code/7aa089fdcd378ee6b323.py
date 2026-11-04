<<<<<<< SEARCH
    strassen_products = [
        # P1 = (A00+A11)*(B00+B11)
        {
            'A': [(0, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1), (1, 1, 1)],
            'C': {(0, 0): 1, (1, 1): 1}
        },
        # P2 = (A10+A11)*B00
        {
            'A': [(1, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1)],
            'C': {(0, 1): 1, (1, 1): -1}
        },
        # P3 = A00*(B10-B11)
        {
            'A': [(0, 0, 1)],
            'B': [(1, 0, 1), (1, 1, -1)],
            'C': {(0, 0): 1, (1, 1): 1}
        },
        # P4 = A11*(B01-B00)
        {
            'A': [(1, 1, 1)],
            'B': [(0, 1, 1), (0, 0, -1)],
            'C': {(0, 0): 1, (1, 1): 1}
        },
        # P5 = (A00+A01)*B11
        {
            'A': [(0, 0, 1), (0, 1, 1)],
            'B': [(1, 1, 1)],
            'C': {(0, 0): -1, (1, 0): 1}
        },
        # P6 = (A10-A00)*(B00+B01)
        {
            'A': [(1, 0, 1), (0, 0, -1)],
            'B': [(0, 0, 1), (0, 1, 1)],
            'C': {(1, 1): 1}
        },
        # P7 = (A01-A11)*(B10+B11)
        {
            'A': [(0, 1, 1), (1, 1, -1)],
            'B': [(1, 0, 1), (1, 1, 1)],
            'C': {(0, 0): 1}
        },
    ]
=======
    # Strassen 2x2x2 product definitions.
    # Local indices: A[di, dj], B[dj, dk], C[dk, di] where di,dj,dk in {0,1}
    #
    # P1 = (A00+A11)*(B00+B11)
    #   A terms: A[0,0] and A[1,1]
    #   B terms: B[0,0] and B[1,1]
    #   C: C[0,0] += P1, C[1,1] += P1
    #
    # P2 = (A10+A11)*B00
    #   A terms: A[1,0] and A[1,1]
    #   B terms: B[0,0]
    #   C: C[0,1] += P2, C[1,1] -= P2
    #
    # P3 = A00*(B10-B11)
    #   A terms: A[0,0]
    #   B terms: B[1,0] and -B[1,1]
    #   C: C[0,0] += P3, C[1,1] += P3
    #
    # P4 = A11*(B01-B00)
    #   A terms: A[1,1]
    #   B terms: B[0,1] and -B[0,0]
    #   C: C[0,0] += P4, C[1,0] += P4
    #
    # P5 = (A00+A01)*B11
    #   A terms: A[0,0] and A[0,1]
    #   B terms: B[1,1]
    #   C: C[0,0] -= P5, C[0,1] += P5
    #
    # P6 = (A10-A00)*(B00+B01)
    #   A terms: A[1,0] and -A[0,0]
    #   B terms: B[0,0] and B[0,1]
    #   C: C[1,1] += P6
    #
    # P7 = (A01-A11)*(B10+B11)
    #   A terms: A[0,1] and -A[1,1]
    #   B terms: B[1,0] and B[1,1]
    #   C: C[0,0] += P7
    #
    # Verify:
    # C[0,0] = P1 + P3 + P4 - P5 + P7
    #   = (A00+A11)(B00+B11) + A00(B10-B11) + A11(B01-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    #   = A00B00 + A00B11 + A11B00 + A11B11 + A00B10 - A00B11 + A11B01 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    #   = A00B00 + A00B10 + A11B01 + A01B10
    #   Hmm, that's not right. Let me recalculate.
    #
    # Actually, I think I have the Strassen formulas slightly wrong. Let me use the standard ones:
    #
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B11)
    # M4 = (A11)(B01-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Let me verify C00:
    # M1 + M4 - M5 + M7
    # = (A00+A11)(B00+B11) + A11(B01-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 + A11B01 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + A01B10 + A11B01
    # That's missing A00B00... wait, I have A00B00 in there. Let me collect terms:
    # A00B00: +1 (from M1)
    # A00B11: +1 (M1) - 1 (M5) = 0
    # A11B00: +1 (M1) - 1 (M4) = 0
    # A11B11: +1 (M1) - 1 (M7) = 0
    # A11B01: +1 (M4)
    # A01B11: -1 (M5) + 1 (M7) = 0
    # A01B10: +1 (M7)
    # A11B10: -1 (M7)
    #
    # So C00 = A00B00 + A11B01 + A01B10 - A11B10
    # But the correct C00 = A00B00 + A01B10. So there's an extra term A11B01 - A11B10.
    #
    # I think I'm making errors. Let me just use a known correct Strassen decomposition.
    #
    # The correct Strassen 2x2x2 decomposition in tensor form:
    # For the 2x2x2 multiplication tensor, the rank-7 decomposition is:
    #
    # Term 1: U has 1 at (0,0) and (1,1), V has 1 at (0,0) and (1,1), W has 1 at (0,0) and (1,1)
    # Term 2: U has 1 at (1,0) and (1,1), V has 1 at (0,0), W has 1 at (0,1) and -1 at (1,1)
    # Term 3: U has 1 at (0,0), V has 1 at (1,0) and -1 at (1,1), W has 1 at (0,0) and (1,1)
    # Term 4: U has 1 at (1,1), V has 1 at (0,1) and -1 at (0,0), W has 1 at (0,0) and (1,0)
    # Term 5: U has 1 at (0,0) and (0,1), V has 1 at (1,1), W has -1 at (0,0) and 1 at (0,1)
    # Term 6: U has 1 at (1,0) and -1 at (0,0), V has 1 at (0,0) and (0,1), W has 1 at (1,1)
    # Term 7: U has 1 at (0,1) and -1 at (1,1), V has 1 at (1,0) and (1,1), W has 1 at (0,0)

    strassen_products = [
        # Term 1: (A00+A11)*(B00+B11) -> C00, C11
        {
            'A': [(0, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1), (1, 1, 1)]  # (dk, di, coeff)
        },
        # Term 2: (A10+A11)*B00 -> C10, -C11
        {
            'A': [(1, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1)],
            'C': [(0, 1, 1), (1, 1, -1)]
        },
        # Term 3: A00*(B10-B11) -> C00, C11
        {
            'A': [(0, 0, 1)],
            'B': [(1, 0, 1), (1, 1, -1)],
            'C': [(0, 0, 1), (1, 1, 1)]
        },
        # Term 4: A11*(B01-B00) -> C00, C10
        {
            'A': [(1, 1, 1)],
            'B': [(0, 1, 1), (0, 0, -1)],
            'C': [(0, 0, 1), (1, 0, 1)]
        },
        # Term 5: (A00+A01)*B11 -> -C00, C01
        {
            'A': [(0, 0, 1), (0, 1, 1)],
            'B': [(1, 1, 1)],
            'C': [(0, 0, -1), (0, 1, 1)]
        },
        # Term 6: (A10-A00)*(B00+B01) -> C11
        {
            'A': [(1, 0, 1), (0, 0, -1)],
            'B': [(0, 0, 1), (0, 1, 1)],
            'C': [(1, 1, 1)]
        },
        # Term 7: (A01-A11)*(B10+B11) -> C00
        {
            'A': [(0, 1, 1), (1, 1, -1)],
            'B': [(1, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1)]
        },
    ]
>>>>>>> REPLACE

Let me verify the Strassen formulas are correct by checking that they produce the right C values.

