"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strategy: Decompose 2x12 multiplication into 4 independent 2x3 multiplications.
    # A (2x2) * B (2x12) = C (2x12)
    # Split B into 4 blocks of 2x3: B = [B0, B1, B2, B3]
    # C = [C0, C1, C2, C3] where Ci = A * Bi
    
    # For each 2x2 by 2x3 multiplication, use a rank-5 decomposition.
    # Total rank = 4 * 5 = 20
    
    # Rank-5 decomposition for 2x2 by 2x3:
    # A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    
    # The 12 terms of schoolbook are:
    # c00 = ae + bh
    # c01 = af + bi  
    # c02 = ag + bj
    # c10 = ce + dh
    # c11 = cf + di
    # c12 = cg + dj
    
    # We can use the following 5-term decomposition (verified):
    # Using linear combinations to reduce from 12 to 5 terms.
    
    # Let's define the 5 rank-1 tensors for 2x2 x 2x3:
    # A vectors (4 components: a, b, c, d)
    # B vectors (6 components: e, f, g, h, i, j)
    # C vectors (6 components: c00, c01, c02, c10, c11, c12)
    
    A5 = [
        [1, 0, 0, 0],   # a
        [0, 1, 0, 0],   # b
        [1, 1, 0, 0],   # a+b
        [0, 0, 1, 0],   # c
        [0, 0, 1, 1],   # c+d
    ]
    
    B5 = [
        [1, 0, 0, 1, 0, 0],   # e+h
        [0, 1, 0, 0, 1, 0],   # f+i
        [0, 0, 1, 0, 0, 1],   # g+j
        [1, 0, 0, 0, 0, 0],   # e
        [0, 0, 0, 1, 0, 0],   # h
    ]
    
    C5 = [
        [1, 0, 0, 0, 0, 0],   # c00
        [0, 1, 0, 0, 0, 0],   # c01
        [0, 0, 1, 0, 0, 0],   # c02
        [0, 0, 0, 1, 0, 0],   # c10
        [0, 0, 0, 0, 1, 0],   # c11
    ]
    
    # This doesn't work directly. Let me use a proper construction.
    # 
    # Actually, for 2x2 x 2x3, the minimal rank is known to be 7 (not 5).
    # Let me use rank 7 for each block, giving total rank 28.
    # But wait, let me try to find a better decomposition.
    
    # Actually, let me just use the schoolbook approach with rank 24 (4 blocks * 6 terms each).
    # This is safe and correct.
    
    rank = n * m * p  # 24
    U = [[0] * rank for _ in range(n * m)]  # 4 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 24 x 24
    
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
