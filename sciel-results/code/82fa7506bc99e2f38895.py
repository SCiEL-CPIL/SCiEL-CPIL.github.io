"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 * 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We map the 4 middle indices to 2x2 blocks.
    # A (2x4) -> A_block (2x2) of vectors (1x2)
    # B (4x8) -> B_block (2x2) of matrices (2x4)
    
    # We assume the decomposition T_2,4,8 = T_2,2,2 (Strassen) x T_1,2,2 (Rank 4)
    # This requires mapping the 8 output columns to 4 groups of 2.
    
    # Strassen A coefficients (2x2)
    # Indices: 00, 01, 10, 11
    strassen_A = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 0, 1, 0], # M5
        [0, 0, 1, -1], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # Strassen B coefficients (2x2)
    strassen_B = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1], # M3
        [0, 1, -1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Rank 4 decomposition for 1x2 * 2x2
    # C0 = a0*b00 + a1*b10
    # C1 = a0*b01 + a1*b11
    # Terms:
    # 0: a0 * b00 -> C0
    # 1: a1 * b10 -> C0
    # 2: a0 * b01 -> C1
    # 3: a1 * b11 -> C1
    # U coeffs for a0, a1
    # V coeffs for b00, b10, b01, b11
    
    # We have 4 column pairs in B (width 8 -> 4 pairs of 2)
    # For each Strassen multiplier (7) and each column pair (4), we have 4 terms?
    # Total 7 * 4 * 4 = 112 terms?
    # No, the rank is 28.
    # This implies we only have 7 * 4 = 28 terms.
    # This means the "Rank 4" part must be folded into the 7?
    # Or the "4" column pairs are handled by the 7?
    
    # Given the ambiguity and risk of error in manual tensor construction,
    # and the strict requirement for exact decomposition,
    # I will fall back to the safe Schoolbook rank 64 if I cannot verify the 28-rank construction.
    # However, the directive asks for the 28-rank approach.
    # I will implement the 28-rank construction assuming the structure holds.
    
    col_idx = 0
    # 7 Strassen multipliers
    for s in range(7):
        # 4 column pairs (0-1, 2-3, 4-5, 6-7)
        for c_pair in range(4):
            # 1 term per (s, c_pair) ?
            # If rank is 28, then 1 term per (s, c_pair).
            # This implies the 1x2 * 2x2 product is rank 1? Impossible.
            
            # If the rank is 28, and we have 7 Strassen steps,
            # each step must handle the 4 column pairs with rank 4?
            # 7 * 4 = 28.
            # So for each Strassen multiplier, we add 4 terms.
            
            for t in range(4):
                # t corresponds to the 4 terms of the 1x2 * 2x2 decomposition
                # But we only have 1 column index available in the 28 rank budget?
                # No, 28 is the total rank.
                
                # Let's assume the 4 terms of the sub-decomposition are the 4 columns of c_pair?
                # No, c_pair is a pair of columns.
                
                # This construction is too complex to derive correctly without verification.
                # I will return the schoolbook solution to ensure validity.
                pass

    # Fallback to Schoolbook
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
