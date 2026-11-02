"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We construct a rank-20 decomposition by exploiting the 4x4 tensor structure.
    # The idea: embed 3x3 into 4x4, use a rank-20 decomposition of 4x4,
    # then restrict to indices 0,1,2.
    
    # Actually, let's build it directly with careful term sharing.
    # Partition: A = [[A22, a], [aT, s]] where A22 is 2x2, a is 2x1, aT is 1x2, s is scalar
    # B = [[B22, b], [bT, t]] where B22 is 2x2, b is 2x1, bT is 1x2, t is scalar
    
    # C = A*B has blocks:
    # C11 = A22*B22 + a*bT
    # C12 = A22*b + a*t
    # C21 = aT*B22 + s*bT
    # C22 = aT*b + s*t
    
    # Strategy: Use 7 terms for A22*B22 (Strassen), and share terms for the rest.
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for A22*B22 (7 terms)
    # U indices: A00=0, A01=1, A10=3, A11=4
    # V indices: B00=0, B01=1, B10=3, B11=4
    # W indices: C00=0, C01=3, C10=1, C11=4
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # Now handle boundary terms more efficiently.
    # a = [A02, A12] -> U indices 2, 5
    # aT = [A20, A21] -> U indices 6, 7
    # s = A22 -> U index 8
    # b = [B02, B12] -> V indices 2, 5
    # bT = [B20, B21] -> V indices 6, 7
    # t = B22 -> V index 8
    
    # C11 needs: a*bT = A02*B20 + A02*B21 + A12*B20 + A12*B21
    # But we can share: (A02+A12)(B20+B21) gives cross terms we need to subtract
    # Let's use: M8 = (A02+A12)(B20+B21)
    # Then C00 += M8 - A02*B21 - A12*B20
    # C11 += M8 - A02*B21 - A12*B20 ... this gets complicated.
    
    # Simpler: just add the 4 terms for a*bT, 4 for A22*b, 2 for a*t,
    # 4 for aT*B22, 2 for s*bT, 2 for aT*b, 1 for s*t
    # That's 4+4+2+4+2+2+1 = 19 + 7 = 26. Too many.
    
    # Let's be smarter. Combine some terms.
    
    # For C12 = A22*b + a*t:
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # Use: M8 = (A00+A01+A02)(B02) ... no, let's just do it directly.
    
    # C02: 3 terms, C12: 3 terms -> 6 terms
    # C21: 3 terms, C22: 3 terms -> 6 terms  
    # a*bT: 4 terms
    # Total: 7 + 6 + 6 + 4 = 23
    
    # Can we do better? Let's try to share.
    
    # Actually, let me just implement rank 23 first and verify it works.
    
    # C02 = A00*B02 + A01*B12 + A02*B22
    U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
    U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
    U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
    
    # C12 = A10*B02 + A11*B12 + A12*B22
    U[3][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
    U[4][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1
    U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
    
    # C20 = A20*B00 + A21*B10 + A22*B20
    U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
    U[7][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
    U[8][col] = 1; V[6][col] = 1; W[2][col] = 1; col += 1
    
    # C21 = A20*B01 + A21*B11 + A22*B21
    U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
    U[7][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1
    U[8][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1
    
    # C22 = A20*B02 + A21*B12 + A22*B22
    U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
    U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
    
    # a*bT for C11: A02*B20 + A02*B21 + A12*B20 + A12*B21
    # Use (A02+A12)(B20+B21) and subtract cross terms
    # M = (A02+A12)(B20+B21) = A02*B20 + A02*B21 + A12*B20 + A12*B21
    # This is exactly what we need for C11!
    U[2][col] = 1; U[5][col] = 1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = 1
    col += 1
    
    # Wait, that's only 1 term for 4 products! Let me verify:
    # (A02+A12)(B20+B21) = A02*B20 + A02*B21 + A12*B20 + A12*B21
    # Yes! So C11 gets this added.
    
    # But wait, C11 also gets contributions from the Strassen terms.
    # Let me check: the Strassen terms already handle A22*B22.
    # The a*bT part adds to C11.
    
    # So total: 7 (Strassen) + 3+3 (C02,C12) + 3+3 (C20,C21) + 3 (C22) + 1 (a*bT) = 23
    
    # Let me verify col = 23
    # We used: 7 + 3 + 3 + 3 + 3 + 3 + 1 = 23. Yes.
    
    # But rank is declared as 20. Let me fix that.
    # Actually, I should just set rank = col at the end and trim.
    
    # Let me restructure: build all terms, then trim unused columns.
    
    # Reset and redo properly
    U = [[0] * 30 for _ in range(n * m)]
    V = [[0] * 30 for _ in range(m * p)]
    W = [[0] * 30 for _ in range(p * n)]
    
    col = 0
    
    # Strassen for A22*B22
    U[0][col] = 1; U[4][col] = 1; V[0][col] = 1; V[4][col] = 1; W[0][col] = 1; W[4][col] = 1; col += 1
    U[3][col] = 1; U[4][col] = 1; V[0][col] = 1; W[1][col] = 1; W[4][col] = -1; col += 1
    U[0][col] = 1; V[1][col] = 1; V[4][col] = -1; W[3][col] = 1; W[4][col] = 1; col += 1
    U[4][col] = 1; V[3][col] = 1; V[0][col] = -1; W[0][col] = 1; W[1][col] = 1; col += 1
    U[0][col] = 1; U[1][col] = 1; V[4][col] = 1; W[0][col] = -1; W[3][col] = 1; col += 1
    U[3][col] = 1; U[0][col] = -1; V[0][col] = 1; V[1][col] = 1; W[4][col] = 1; col += 1
    U[1][col] = 1; U[4][col] = -1; V[3][col] = 1; V[4][col] = 1; W[0][col] = 1; col += 1
    
    # C02
    U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
    U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
    U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
    
    # C12
    U[3][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
    U[4][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1
    U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
    
    # C20
    U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
    U[7][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
    U[8][col] = 1; V[6][col] = 1; W[2][col] = 1; col += 1
    
    # C21
    U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
    U[7][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1
    U[8][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1
    
    # C22
    U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
    U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
    
    # a*bT for C11
    U[2][col] = 1; U[5][col] = 1; V[6][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
    
    # Trim to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
