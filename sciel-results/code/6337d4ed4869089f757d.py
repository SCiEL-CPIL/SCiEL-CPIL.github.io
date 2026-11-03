def _strassen_4x4_4x4():
    """
    Rank-49 decomposition for 4x4 * 4x4.
    Returns U, V, W of shape (16, 49).
    U rows: i*4+j
    V rows: j*4+k
    W rows: k*4+i
    """
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Strassen 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We map 4x4 to 2x2 blocks of 2x2.
    # A blocks: A00 (0-1, 0-1), A01 (0-1, 2-3), A10 (2-3, 0-1), A11 (2-3, 2-3)
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # For each Strassen term t (0-6), we have a linear combination of A blocks and B blocks.
    # We need to expand this to the 2x2 level.
    
    # Let's define the 2x2 Strassen factors first.
    # U2, V2, W2 for 2x2x2.
    U2 = [
        [1, 0, 0, 0, 0, 0, 0], # A00
        [0, 0, 1, 0, 0, 0, 0], # A01
        [0, 1, 0, 0, 1, 1, 0], # A10
        [0, 0, 0, 1, 0, -1, 1] # A11
    ]
    # Wait, the standard Strassen U/V/W mapping needs to be consistent.
    # Let's use the explicit terms from the seed, which seemed correct in structure.
    
    # I will reuse the seed's _make_strassen_2x2 but ensure it's correct.
    # Seed:
    # M1: U: A00+A11, V: B00+B11, W: C00+C11-C10? No.
    # Let's trust the seed's _make_strassen_2x2 and _strassen_4x4_4x4 logic, 
    # but the issue might be in the assembly in construct().
    
    return U, V, W # Placeholder
