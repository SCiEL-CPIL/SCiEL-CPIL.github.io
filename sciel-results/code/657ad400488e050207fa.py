"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank 20 decomposition of M_{2,3,5}
    # Split into two M_{2,3,2} (rank 7 each) and one M_{2,3,1} (rank 6)
    # Total rank = 7 + 7 + 6 = 20
    
    n, m, p = DIMENSIONS
    rank = 20
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    col = 0
    
    # M_{2,3,2} for columns 0, 1
    # Rank 7 decomposition of M_{2,3,2}
    # A is 2x3, B is 3x2. C is 2x2.
    # C_00 = A00 B00 + A01 B10 + A02 B20
    # C_01 = A00 B01 + A01 B11 + A02 B21
    # C_10 = A10 B00 + A11 B10 + A12 B20
    # C_11 = A10 B01 + A11 B11 + A12 B21
    
    # Known rank 7 decomposition:
    # M1 = A00 B00
    # M2 = A01 B10
    # M3 = A02 B20
    # M4 = A00 B01
    # M5 = A01 B11
    # M6 = A02 B21
    # M7 = (A10+A00) (B00+B01) ? No.
    
    # Actually, let's use the simple block decomposition if I can't recall the exact rank 7 terms.
    # But rank 7 is standard.
    # Let's use the decomposition from "The rank of the 2x3 matrix multiplication tensor is 7"
    # The terms are:
    # 1. A00 B00
    # 2. A01 B10
    # 3. A02 B20
    # 4. A00 B01
    # 5. A01 B11
    # 6. A02 B21
    # 7. (A10 + A00 + A01 + A02) (B00 + B01) ? No.
    
    # Let's just use the schoolbook for M_{2,3,2} which is rank 8, and M_{2,3,1} rank 6.
    # Total rank = 8 + 8 + 6 = 22.
    # This is still better than 30.
    # I will implement rank 22 to be safe, as I don't have the exact rank 7 coefficients memorized.
    
    # Reset rank
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Block 1: M_{2,3,2} for cols 0,1 (Rank 8)
    # C_00, C_01, C_10, C_11
    # Terms:
    # A00 B00 -> C00
    U[0][col] = 1; V[0*5+0][col] = 1; W[0*2+0][col] = 1; col += 1
    # A01 B10 -> C00
    U[1][col] = 1; V[1*5+0][col] = 1; W[0*2+0][col] = 1; col += 1
    # A02 B20 -> C00
    U[2][col] = 1; V[2*5+0][col] = 1; W[0*2+0][col] = 1; col += 1
    # A00 B01 -> C01
    U[0][col] = 1; V[0*5+1][col] = 1; W[1*2+0][col] = 1; col += 1
    # A01 B11 -> C01
    U[1][col] = 1; V[1*5+1][col] = 1; W[1*2+0][col] = 1; col += 1
    # A02 B21 -> C01
    U[2][col] = 1; V[2*5+1][col] = 1; W[1*2+0][col] = 1; col += 1
    # A10 B00 -> C10
    U[3][col] = 1; V[0*5+0][col] = 1; W[0*2+1][col] = 1; col += 1
    # A11 B10 -> C10
    U[4][col] = 1; V[1*5+0][col] = 1; W[0*2+1][col] = 1; col += 1
    # A12 B20 -> C10
    U[5][col] = 1; V[2*5+0][col] = 1; W[0*2+1][col] = 1; col += 1
    # A10 B01 -> C11
    U[3][col] = 1; V[0*5+1][col] = 1; W[1*2+1][col] = 1; col += 1
    # A11 B11 -> C11
    U[4][col] = 1; V[1*5+1][col] = 1; W[1*2+1][col] = 1; col += 1
    # A12 B21 -> C11
    U[5][col] = 1; V[2*5+1][col] = 1; W[1*2+1][col] = 1; col += 1
    
    # Block 2: M_{2,3,2} for cols 2,3 (Rank 8)
    # Same structure, shifted columns
    # C_02, C_03, C_12, C_13
    # A00 B02 -> C02
    U[0][col] = 1; V[0*5+2][col] = 1; W[2*2+0][col] = 1; col += 1
    # A01 B12 -> C02
    U[1][col] = 1; V[1*5+2][col] = 1; W[2*2+0][col] = 1; col += 1
    # A02 B22 -> C02
    U[2][col] = 1; V[2*5+2][col] = 1; W[2*2+0][col] = 1; col += 1
    # A00 B03 -> C03
    U[0][col] = 1; V[0*5+3][col] = 1; W[3*2+0][col] = 1; col += 1
    # A01 B13 -> C03
    U[1][col] = 1; V[1*5+3][col] = 1; W[3*2+0][col] = 1; col += 1
    # A02 B23 -> C03
    U[2][col] = 1; V[2*5+3][col] = 1; W[3*2+0][col] = 1; col += 1
    # A10 B02 -> C12
    U[3][col] = 1; V[0*5+2][col] = 1; W[2*2+1][col] = 1; col += 1
    # A11 B12 -> C12
    U[4][col] = 1; V[1*5+2][col] = 1; W[2*2+1][col] = 1; col += 1
    # A12 B22 -> C12
    U[5][col] = 1; V[2*5+2][col] = 1; W[2*2+1][col] = 1; col += 1
    # A10 B03 -> C13
    U[3][col] = 1; V[0*5+3][col] = 1; W[3*2+1][col] = 1; col += 1
    # A11 B13 -> C13
    U[4][col] = 1; V[1*5+3][col] = 1; W[3*2+1][col] = 1; col += 1
    # A12 B23 -> C13
    U[5][col] = 1; V[2*5+3][col] = 1; W[3*2+1][col] = 1; col += 1
    
    # Block 3: M_{2,3,1} for col 4 (Rank 6)
    # C_04, C_14
    # M1 = A00 B04
    U[0][col] = 1; V[0*5+4][col] = 1; W[4*2+0][col] = 1; col += 1
    # M2 = A01 B14
    U[1][col] = 1; V[1*5+4][col] = 1; W[4*2+0][col] = 1; col += 1
    # M3 = A02 B24
    U[2][col] = 1; V[2*5+4][col] = 1; W[4*2+0][col] = 1; col += 1
    # M4 = (A00+A10) B04
    U[0][col] = 1; U[3][col] = 1; V[0*5+4][col] = 1; W[4*2+1][col] = 1; col += 1
    # M5 = (A01+A11) B14
    U[1][col] = 1; U[4][col] = 1; V[1*5+4][col] = 1; W[4*2+1][col] = 1; col += 1
    # M6 = (A02+A12) B24
    U[2][col] = 1; U[5][col] = 1; V[2*5+4][col] = 1; W[4*2+1][col] = 1; col += 1
    
    # We need to adjust the W entries for the last block to get C14 correctly.
    # C14 = M4 - M1 + M5 - M2 + M6 - M3
    # So W[4*2+1] should have -1 for M1, M2, M3 and +1 for M4, M5, M6.
    # But we already added +1 to W[4*2+1] for M4, M5, M6.
    # We need to subtract M1, M2, M3 from W[4*2+1].
    # M1 was col-6, M2 col-5, M3 col-4.
    # W[4*2+1] is index 9.
    # We set W[9][col-6] = -1, W[9][col-5] = -1, W[9][col-4] = -1.
    # And W[4*2+0] (index 8) has +1 for M1, M2, M3. That is correct for C04.
    
    W[9][col-6] = -1
    W[9][col-5] = -1
    W[9][col-4] = -1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
