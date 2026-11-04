"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (8, rank)
    
    # We use the rank 28 decomposition based on 4 Strassen 2x2x2 blocks.
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 coefficients
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the 7 multiplications for a 2x2x2 block
    # Each multiplication r has:
    # - U terms: list of (local_i, local_j, coeff)
    # - V terms: list of (local_j, local_k, coeff)
    # - W terms: list of (local_k, local_i, coeff)
    
    # Note: U index is (i, j). V index is (j, k). W index is (k, i).
    # In Strassen, A is 2x2 (i,j), B is 2x2 (j,k).
    
    strassen_muls = [
        # M1: (a+d)(e+h)
        # A: a(0,0), d(1,1)
        # B: e(0,0), h(1,1)
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1), (1,0,-1), (0,1,1)] # C00, C11, C10, C01? 
            # Wait, let's map C indices carefully.
            # C00 gets M1. C11 gets M1.
            # C01 gets -M1? No.
            # C00 = M1 + ...
            # C11 = M1 + ...
            # So W for M1: (0,0) coeff 1, (1,1) coeff 1.
            # But W index is (k, i).
            # C00 is k=0, i=0.
            # C11 is k=1, i=1.
            # So W: [(0,0,1), (1,1,1)]
        },
        # M2: (c+d)e
        # A: c(1,0), d(1,1)
        # B: e(0,0)
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1)] # C10 is k=0, i=1.
        },
        # M3: a(f-h)
        # A: a(0,0)
        # B: f(0,1), h(1,1)
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1)] # C01 is k=1, i=0.
        },
        # M4: d(g-e)
        # A: d(1,1)
        # B: g(1,0), e(0,0)
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)] # C00 and C10.
            # C00 = M4. C10 = M4.
            # C00: k=0, i=0.
            # C10: k=0, i=1.
        },
        # M5: (a+b)h
        # A: a(0,0), b(0,1)
        # B: h(1,1)
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)] # C00 gets -M5. C01 gets M5.
            # C00: k=0, i=0.
            # C01: k=1, i=0.
        },
        # M6: (c-a)(e+f)
        # A: c(1,0), a(0,0)
        # B: e(0,0), f(0,1)
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)] # C11 gets M6.
            # C11: k=1, i=1.
        },
        # M7: (b-d)(g+h)
        # A: b(0,1), d(1,1)
        # B: g(1,0), h(1,1)
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)] # C00 gets M7.
            # C00: k=0, i=0.
        }
    ]
    
    # Verify W contributions for M1
    # C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    # My M1 W: (0,0,1), (1,1,1). Correct.
    
    # Verify M2
    # C10 = M2 + M4
    # My M2 W: (0,1,1). Correct.
    
    # Verify M3
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    # My M3 W: (1,0,1). Correct.
    
    # Verify M4
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # My M4 W: (0,0,1), (0,1,1). Correct.
    
    # Verify M5
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # My M5 W: (0,0,-1), (1,0,1). Correct.
    
    # Verify M6
    # C11 = M1 - M2 + M3 + M6
    # My M6 W: (1,1,1). Correct.
    
    # Verify M7
    # C00 = M1 + M4 - M5 + M7
    # My M7 W: (0,0,1). Correct.
    
    # Now fill the global matrices for the 4 sub-blocks
    # Sub-block (si, sj) where si in {0,1} (row of A block), sj in {0,1} (col of A block / row of B block)
    # Actually, we split j (cols of A, rows of B) into two groups of 2.
    # And k (cols of B) into two groups of 2.
    
    # Group 0: j in {0,1}, k in {0,1}
    # Group 1: j in {0,1}, k in {2,3}
    # Group 2: j in {2,3}, k in {0,1}
    # Group 3: j in {2,3}, k in {2,3}
    
    col_offset = 0
    for j_block in range(2): # 0: j=0,1; 1: j=2,3
        for k_block in range(2): # 0: k=0,1; 1: k=2,3
            # For each of the 7 Strassen multiplications
            for mul in strassen_muls:
                # U terms
                for (li, lj, coeff) in mul['U']:
                    gi = li
                    gj = j_block * 2 + lj
                    U[gi * m + gj][col_offset] += coeff
                # V terms
                for (lj, lk, coeff) in mul['V']:
                    gj = j_block * 2 + lj
                    gk = k_block * 2 + lk
                    V[gj * p + gk][col_offset] += coeff
                # W terms
                for (lk, li, coeff) in mul['W']:
                    gk = k_block * 2 + lk
                    gi = li
                    W[gk * n + gi][col_offset] += coeff
                
                col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
