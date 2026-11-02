"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try rank 24: 4 blocks with shared Strassen terms
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We use a modified approach: for each of the 4 column pairs,
    # we use 6 independent Strassen-like terms, but share 4 terms across blocks.
    # Actually, let's try a direct rank-24 construction.
    
    # Standard Strassen for 2x2x2 has rank 7.
    # For 2x2x8, naive is 28. We seek rank 24.
    
    # Let's use 4 blocks, each with 7 terms = 28, but try to merge.
    # Instead, let's try a different decomposition.
    
    # Approach: Use 6 terms per block for 4 blocks = 24, 
    # but this won't work since 2x2x2 needs rank >= 7.
    
    # Alternative: Use the fact that we can have non-integer coefficients.
    # Let me try a rank-24 construction based on known results.
    
    # Actually, the minimal rank for 2x2x8 is known to be 24.
    # Let me construct it explicitly.
    
    # Use 4 groups of 6 terms each? No, that's only 24 but 2x2x2 needs 7.
    
    # Let me think differently. The tensor M_{2,2,8} can be seen as 
    # 4 copies of M_{2,2,2} but with shared A.
    
    # A better approach: use 28 terms but find 4 linear dependencies.
    # Or construct directly with rank 24.
    
    # Let me use a known rank-24 construction for 2x2x8.
    # Based on the structure, we can use:
    # - 4 blocks, each contributing some terms
    # - With 4 shared terms
    
    # Let me try: 4*6 + 4 = 28... no.
    # 3*8 = 24? Or 4*6 = 24?
    
    # Let me just try to build a valid rank-24 decomposition.
    
    # I'll use a systematic approach:
    # For 2x2x8, we can view it as multiplying A (2x2) with 4 independent 2x2 blocks of B.
    # The key is that A is shared, so we can factor out A's contributions.
    
    # Let me use the following construction with rank 24:
    # Terms 0-17: 4 blocks x 4 terms each (basic products)
    # Terms 18-23: 6 shared Strassen-like correction terms
    
    # Actually, let me try a cleaner approach.
    
    # Use 28 terms but set up linear dependencies to reduce effective rank.
    # No, we need actual rank reduction.
    
    # Let me try: for each of the 4 column pairs, use 6 terms, total 24.
    # But 2x2x2 requires rank 7, so 6 terms per block won't work independently.
    
    # The trick: the 4 blocks are not independent because they share A.
    
    # Let me construct a rank-24 solution:
    # Use the following 24 bilinear forms:
    
    # I'll use a known construction. For 2x2x8, rank 24 is achievable.
    
    # Let me define the 24 terms explicitly.
    
    # Terms for block q (q=0,1,2,3), columns c0=2q, c1=2q+1:
    # We'll use 6 terms per block, but with cross-block sharing.
    
    # Actually, let me just use 28 terms but verify correctness first,
    # then try to reduce.
    
    # For now, let me return the rank-28 solution as a baseline,
    # and I'll need to find a rank-24 construction.
    
    # Let me try a different approach: use the fact that we can have
    # half-integer coefficients and find a rank-24 decomposition.
    
    # Known result: rank(2,2,8) = 24.
    
    # Construction:
    # Let me use 4 blocks. For each block, instead of 7 Strassen terms,
    # use 6 terms, but add 4 cross-block terms.
    
    # Total: 4*6 + 4 = 28. Not 24.
    
    # Let me try: 3*8 = 24.
    
    # I'll use a different decomposition strategy.
    
    # For now, let me just return a valid rank-28 solution and note that
    # we need to find rank 24.
    
    # Let me try to actually construct rank 24.
    
    # Based on literature, here's a rank-24 construction for 2x2x8:
    
    # I'll use the following approach:
    # - 4 independent 2x2x2 multiplications would need 28
    # - But since A is shared, we can save 4 multiplications
    
    # Let me define the 24 terms:
    
    for q in range(4):
        c0 = 2 * q
        c1 = 2 * q + 1
        
        # Term 0: A00*B00
        r = 6 * q + 0
        U[0][r] = 1
        V[c0][r] = 1
        W[2 * c0][r] = 1
        
        # Term 1: A00*B01
        r = 6 * q + 1
        U[0][r] = 1
        V[c1][r] = 1
        W[2 * c1][r] = 1
        
        # Term 2: A01*B10
        r = 6 * q + 2
        U[1][r] = 1
        V[8 + c0][r] = 1
        W[2 * c0][r] = 1
        
        # Term 3: A01*B11
        r = 6 * q + 3
        U[1][r] = 1
        V[8 + c1][r] = 1
        W[2 * c1 + 1][r] = 1
        
        # Term 4: A10*B00
        r = 6 * q + 4
        U[2][r] = 1
        V[c0][r] = 1
        W[2 * c0 + 1][r] = 1
        
        # Term 5: A10*B01
        r = 6 * q + 5
        U[2][r] = 1
        V[c1][r] = 1
        W[2 * c1 + 1][r] = 1
    
    # This gives 24 terms but is missing A11*B10 and A11*B11 for each block.
    # We need to add these, but that would exceed rank 24.
    
    # This approach doesn't work. Let me use Strassen properly.
    
    # Let me go back to rank 28 for now.
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for q in range(4):
        c0 = 2 * q
        c1 = 2 * q + 1
        
        # M1 = (A00+A11)(B00+B11)
        r = 7 * q + 0
        U[0][r] = 1
        U[3][r] = 1
        V[c0][r] = 1
        V[8 + c1][r] = 1
        W[2 * c0][r] = 1
        W[2 * c1 + 1][r] = 1
        
        # M2 = (A10+A11)B00
        r = 7 * q + 1
        U[2][r] = 1
        U[3][r] = 1
        V[c0][r] = 1
        W[2 * c0 + 1][r] = 1
        W[2 * c1 + 1][r] = -1
        
        # M3 = A00(B01-B11)
        r = 7 * q + 2
        U[0][r] = 1
        V[c1][r] = 1
        V[8 + c1][r] = -1
        W[2 * c1][r] = 1
        W[2 * c1 + 1][r] = 1
        
        # M4 = A11(B10-B00)
        r = 7 * q + 3
        U[3][r] = 1
        V[8 + c0][r] = 1
        V[c0][r] = -1
        W[2 * c0][r] = 1
        W[2 * c0 + 1][r] = 1
        
        # M5 = (A00+A01)B11
        r = 7 * q + 4
        U[0][r] = 1
        U[1][r] = 1
        V[8 + c1][r] = 1
        W[2 * c0][r] = -1
        W[2 * c1][r] = 1
        
        # M6 = (A10-A00)(B00+B01)
        r = 7 * q + 5
        U[2][r] = 1
        U[0][r] = -1
        V[c0][r] = 1
        V[c1][r] = 1
        W[2 * c1 + 1][r] = 1
        
        # M7 = (A01-A11)(B10+B11)
        r = 7 * q + 6
        U[1][r] = 1
        U[3][r] = -1
        V[8 + c0][r] = 1
        V[8 + c1][r] = 1
        W[2 * c0][r] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
