import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 43 decomposition
    # Part 1: M_3,3,3 (Rank 26)
    # Part 2: M_3,3,2 (Rank 17)
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_idx, v_idx, w_idx are indices in U, V, W
    # val is the coefficient
    def add_term(u_idx, v_idx, w_idx, val, col):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    # Strassen 2x2x2 decomposition
    # Indices for 2x2 block:
    # A: 0,1,2,3 (0:11, 1:12, 2:21, 3:22)
    # B: 0,1,2,3
    # C: 0,1,2,3
    # We map these to the 3x3 indices.
    # For M_3,3,3 part (cols 0-25):
    # A indices: 0-8. B indices: 0-8. C indices: 0-8.
    # We use the top-left 2x2 block for Strassen.
    # A_22 corresponds to A[0:2, 0:2] -> indices 0,1,2,3
    # a corresponds to A[0:2, 2] -> indices 1, 4? No.
    # A is 3x3.
    # A = [[A00, A01, A02],
    #      [A10, A11, A12],
    #      [A20, A21, A22]]
    # A_22 (top-left 2x2): A00, A01, A10, A11. Indices 0, 1, 3, 4.
    # a (top-right 2x1): A02, A12. Indices 2, 5.
    # a^T (bottom-left 1x2): A20, A21. Indices 6, 7.
    # alpha (bottom-right 1x1): A22. Index 8.
    
    # B_22 (top-left 2x2): B00, B01, B10, B11. Indices 0, 1, 3, 4.
    # b (top-right 2x1): B02, B12. Indices 2, 5.
    # b^T (bottom-left 1x2): B20, B21. Indices 6, 7.
    # beta (bottom-right 1x1): B22. Index 8.
    
    # C_22 (top-left 2x2): C00, C01, C10, C11. Indices 0, 1, 3, 4.
    # c (top-right 2x1): C02, C12. Indices 2, 5.
    # c^T (bottom-left 1x2): C20, C21. Indices 6, 7.
    # gamma (bottom-right 1x1): C22. Index 8.

    # Strassen terms for A_22 * B_22 -> C_22
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0, 1, 0) # A00 B00
    add_term(0, 4, 0, 1, 0) # A00 B11
    add_term(4, 0, 0, 1, 0) # A11 B00
    add_term(4, 4, 0, 1, 0) # A11 B11
    # M2 = (A10+A11)B00
    add_term(3, 0, 0, 1, 1) # A10 B00
    add_term(4, 0, 0, 1, 1) # A11 B00
    # M3 = A00(B01-B11)
    add_term(0, 1, 0, 1, 2) # A00 B01
    add_term(0, 4, 0, -1, 2) # A00 B11
    # M4 = A11(B10-B00)
    add_term(4, 3, 0, 1, 3) # A11 B10
    add_term(4, 0, 0, -1, 3) # A11 B00
    # M5 = (A00+A01)B11
    add_term(0, 4, 0, 1, 4) # A00 B11
    add_term(1, 4, 0, 1, 4) # A01 B11
    # M6 = (A10-A00)(B00+B01)
    add_term(3, 0, 0, 1, 5) # A10 B00
    add_term(3, 1, 0, 1, 5) # A10 B01
    add_term(0, 0, 0, -1, 5) # A00 B00
    add_term(0, 1, 0, -1, 5) # A00 B01
    # M7 = (A01-A11)(B10+B11)
    add_term(1, 3, 0, 1, 6) # A01 B10
    add_term(1, 4, 0, 1, 6) # A01 B11
    add_term(4, 3, 0, -1, 6) # A11 B10
    add_term(4, 4, 0, -1, 6) # A11 B11

    # Combine Strassen results into C_22
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # This is handled by the W matrix indices.
    # W indices for C_22: 0, 1, 3, 4.
    # We need to sum the contributions from M1..M7 to the correct W slots.
    # The above add_term calls put 1s in U and V, but W is still 0.
    # We must update W for each Strassen multiplication to point to the correct C entries.
    
    # Reset W for Strassen part? No, we built U and V. Now build W.
    # M1 contributes to C00, C11
    W[0][0] += 1 # C00
    W[4][0] += 1 # C11
    # M2 contributes to C10, C11
    W[3][1] += 1 # C10
    W[4][1] -= 1 # C11
    # M3 contributes to C01, C11
    W[1][2] += 1 # C01
    W[4][2] += 1 # C11
    # M4 contributes to C00, C10
    W[0][3] += 1 # C00
    W[3][3] += 1 # C10
    # M5 contributes to C00, C01
    W[0][4] -= 1 # C00
    W[1][4] += 1 # C01
    # M6 contributes to C11
    W[4][5] += 1 # C11
    # M7 contributes to C00, C11
    W[0][6] += 1 # C00
    W[4][6] += 1 # C11

    # Other terms for M_3,3,3
    # 2. a b^T (4 terms)
    # a: A02 (idx 2), A12 (idx 5). b: B02 (idx 2), B12 (idx 5).
    # C_22 gets contributions.
    # C00 += A02 B02
    add_term(2, 2, 0, 1, 7)
    # C01 += A02 B12
    add_term(2, 5, 1, 1, 8)
    # C10 += A12 B02
    add_term(5, 2, 3, 1, 9)
    # C11 += A12 B12
    add_term(5, 5, 4, 1, 10)

    # 3. A_22 b (4 terms)
    # A_22: 0,1,3,4. b: 2,5.
    # c (C02, C12) gets contributions.
    # C02 += A00 B02 + A01 B12
    add_term(0, 2, 2, 1, 11)
    add_term(1, 5, 2, 1, 12)
    # C12 += A10 B02 + A11 B12
    add_term(3, 2, 5, 1, 13)
    add_term(4, 5, 5, 1, 14)

    # 4. a beta (2 terms)
    # a: 2,5. beta: 8.
    # C02 += A02 B22
    add_term(2, 8, 2, 1, 15)
    # C12 += A12 B22
    add_term(5, 8, 5, 1, 16)

    # 5. a^T B_22 (4 terms)
    # a^T: 6,7. B_22: 0,1,3,4.
    # c^T (C20, C21) gets contributions.
    # C20 += A20 B00 + A21 B10
    add_term(6, 0, 6, 1, 17)
    add_term(7, 3, 6, 1, 18)
    # C21 += A20 B01 + A21 B11
    add_term(6, 1, 7, 1, 19)
    add_term(7, 4, 7, 1, 20)

    # 6. alpha b^T (2 terms)
    # alpha: 8. b^T: 6,7.
    # C20 += A22 B20
    add_term(8, 6, 6, 1, 21)
    # C21 += A22 B21
    add_term(8, 7, 7, 1, 22)

    # 7. a^T b (2 terms)
    # a^T: 6,7. b: 2,5.
    # gamma (C22) gets contributions.
    # C22 += A20 B02 + A21 B12
    add_term(6, 2, 8, 1, 23)
    add_term(7, 5, 8, 1, 24)

    # 8. alpha beta (1 term)
    # C22 += A22 B22
    add_term(8, 8, 8, 1, 25)

    # Part 2: M_3,3,2 (Rank 17)
    # Cols 26-42
    # B is 3x2. Indices for B: 0-5.
    # B_22 (2x2): 0,1,3,4.
    # b^T (1x2): 2,5? No.
    # B is 3x2.
    # B = [[B00, B01],
    #      [B10, B11],
    #      [B20, B21]]
    # B_22: 0,1,3,4.
    # b^T: 2,5. (Row 2, Cols 0,1).
    
    # 1. A_22 B_22 (7 terms)
    # Same Strassen as before, but W indices are for C_22 (top 2x2 of 3x2 output).
    # C is 3x2.
    # C_22: 0,1,3,4.
    # c^T: 2,5? No.
    # C = [[C00, C01],
    #      [C10, C11],
    #      [C20, C21]]
    # C_22: 0,1,3,4.
    # c^T: 2,5. (Row 2, Cols 0,1).
    
    # M1
    add_term(0, 0, 0, 1, 26)
    add_term(0, 4, 0, 1, 26)
    add_term(4, 0, 0, 1, 26)
    add_term(4, 4, 0, 1, 26)
    W[0][26] += 1
    W[4][26] += 1
    # M2
    add_term(3, 0, 0, 1, 27)
    add_term(4, 0, 0, 1, 27)
    W[3][27] += 1
    W[4][27] -= 1
    # M3
    add_term(0, 1, 0, 1, 28)
    add_term(0, 4, 0, -1, 28)
    W[1][28] += 1
    W[4][28] += 1
    # M4
    add_term(4, 3, 0, 1, 29)
    add_term(4, 0, 0, -1, 29)
    W[0][29] += 1
    W[3][29] += 1
    # M5
    add_term(0, 4, 0, 1, 30)
    add_term(1, 4, 0, 1, 30)
    W[0][30] -= 1
    W[1][30] += 1
    # M6
    add_term(3, 0, 0, 1, 31)
    add_term(3, 1, 0, 1, 31)
    add_term(0, 0, 0, -1, 31)
    add_term(0, 1, 0, -1, 31)
    W[4][31] += 1
    # M7
    add_term(1, 3, 0, 1, 32)
    add_term(1, 4, 0, 1, 32)
    add_term(4, 3, 0, -1, 32)
    add_term(4, 4, 0, -1, 32)
    W[0][32] += 1
    W[4][32] += 1

    # 2. a b^T (4 terms)
    # a: 2,5. b^T: 2,5.
    # C00 += A02 B20
    add_term(2, 2, 0, 1, 33)
    # C01 += A02 B21
    add_term(2, 5, 1, 1, 34)
    # C10 += A12 B20
    add_term(5, 2, 3, 1, 35)
    # C11 += A12 B21
    add_term(5, 5, 4, 1, 36)

    # 3. a^T B_22 (4 terms)
    # a^T: 6,7. B_22: 0,1,3,4.
    # C20 += A20 B00 + A21 B10
    add_term(6, 0, 6, 1, 37)
    add_term(7, 3, 6, 1, 38)
    # C21 += A20 B01 + A21 B11
    add_term(6, 1, 7, 1, 39)
    add_term(7, 4, 7, 1, 40)

    # 4. alpha b^T (2 terms)
    # alpha: 8. b^T: 2,5.
    # C20 += A22 B20
    add_term(8, 2, 6, 1, 41)
    # C21 += A22 B21
    add_term(8, 5, 7, 1, 42)

    return {"U": U, "V": V, "W": W}
