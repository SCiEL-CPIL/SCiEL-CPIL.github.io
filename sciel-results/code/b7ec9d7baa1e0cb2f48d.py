"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-49 decomposition via Kronecker product of two Strassen 2x2 decompositions.
    # For 4x4 matrix multiplication, partition into 2x2 blocks and apply Strassen at
    # the block level. Each block multiplication (2x2 x 2x2) uses 7 scalar multiplications.
    # Total: 7 * 7 = 49 scalar multiplications.
    
    n, m, p = DIMENSIONS  # all 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Strassen's 7-term decomposition for 2x2 matrices.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # The 7 products are:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # 
    # C = [[C00, C01], [C10, C11]] where:
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # For the tensor decomposition, each term r has:
    # u_r: coefficients for A entries (a,b,c,d) -> flat indices 0,1,2,3
    # v_r: coefficients for B entries (e,f,g,h) -> flat indices 0,1,2,3
    # w_r: coefficients for C entries (C00,C01,C10,C11) -> flat indices 0,1,2,3
    
    # Strassen terms (u, v, w) for 2x2:
    strassen_terms = [
        # m1 = (a+d)(e+h): u=[1,0,0,1], v=[1,0,0,1]
        # Contributes to C00 (+1), C11 (+1)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # m2 = (c+d)e: u=[0,0,1,1], v=[1,0,0,0]
        # Contributes to C10 (+1), C11 (-1)
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        # m3 = a(f-h): u=[1,0,0,0], v=[0,1,0,-1]
        # Contributes to C01 (+1), C11 (+1)
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        # m4 = d(g-c): u=[0,0,0,1], v=[0,0,1,0]
        # Contributes to C00 (+1), C10 (+1)
        ([0,0,0,1], [0,0,1,0], [1,0,1,0]),
        # m5 = (a+b)h: u=[1,1,0,0], v=[0,0,0,1]
        # Contributes to C00 (-1), C01 (+1)
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        # m6 = (c-a)(e+f): u=[-1,0,1,0], v=[1,1,0,0]
        # Contributes to C11 (+1)
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        # m7 = (b-d)(g+h): u=[0,1,0,-1], v=[0,0,1,1]
        # Contributes to C00 (+1)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    # Verify Strassen decomposition for 2x2:
    # C00 = m1 + m4 - m5 + m7
    # = (a+d)(e+h) + d(g-c) - (a+b)h + (b-d)(g+h)
    # = ae + ah + de + dh + dg - dc - ah - bh + bg + bh - dg - dh
    # = ae + de - dc + bg
    # Hmm, this gives ae + de - dc + bg, but we want ae + bg.
    # There's an error. Let me use the correct Strassen formulas.
    
    # Correct Strassen (verified):
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e  
    # m3 = a(f-h)
    # m4 = d(g-c)  -- this is problematic as noted
    # 
    # Actually, the standard Strassen that works is:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-c)
    # m5 = (a+b)(h)
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Let me verify C00 again more carefully:
    # m1 = (a+d)(e+h) = ae + ah + de + dh
    # m4 = d(g-c) = dg - dc
    # m5 = (a+b)h = ah + bh
    # m7 = (b-d)(g+h) = bg + bh - dg - dh
    # C00 = m1 + m4 - m5 + m7
    # = ae + ah + de + dh + dg - dc - ah - bh + bg + bh - dg - dh
    # = ae + de - dc + bg
    # = a(e) + d(e-c) + bg
    # This is NOT ae + bg unless d(e-c) = 0.
    
    # So the standard Strassen I'm remembering is incorrect. Let me use a different valid 7-term decomposition.
    
    # Actually, I recall now that the correct Strassen uses:
    # m4 = d * g - d * c is not separable. The correct form must be different.
    
    # Let me use a known valid 7-term decomposition. One valid set is:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-c)  -- still problematic
    
    # I'll use the schoolbook rank-64 as a safe fallback, but that's the same as the seed score.
    
    # Actually, let me just use the rank-49 construction by treating 4x4 as 2x2 blocks
    # and using 7 terms per block multiplication, even if I can't verify the exact Strassen
    # coefficients from memory. The key insight is that the Kronecker product of two
    # rank-7 decompositions gives rank 49.
    
    # For simplicity and correctness, I'll use the schoolbook rank-64 decomposition
    # which is guaranteed to be correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
