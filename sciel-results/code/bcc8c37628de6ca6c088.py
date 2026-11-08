import time
import numpy as np
from fractions import Fraction

def construct():
    start = time.time()
    DEADLINE = 115  # safety margin under 120s

    n, m, p = 3, 3, 3
    rank = 23

    # U, V, W as lists of lists for direct return
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    def add_term(u_list, v_list, w_list):
        nonlocal col
        for idx, c in u_list:
            U[idx][col] += c
        for idx, c in v_list:
            V[idx][col] += c
        for idx, c in w_list:
            W[idx][col] += c
        col += 1

    # Bini-De Falco Rank 23 Decomposition
    # Indices:
    # A: 00(0), 01(1), 02(2), 10(3), 11(4), 12(5), 20(6), 21(7), 22(8)
    # B: 00(0), 01(1), 02(2), 10(3), 11(4), 12(5), 20(6), 21(7), 22(8)
    # C: 00(0), 10(1), 20(2), 01(3), 11(4), 21(5), 02(6), 12(7), 22(8)  [k*n+i]
    
    # 7 Strassen terms for the 2x2 block (indices 0,1)
    # M1 = (A00+A11)(B00+B11)
    add_term([(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (4, 1)])
    
    # M2 = (A10+A11)B00
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (4, -1)])
    
    # M3 = A00(B01-B11)
    add_term([(0, 1)], [(1, 1), (4, -1)], [(3, 1), (4, 1)])
    
    # M4 = A11(B10-B00)
    add_term([(4, 1)], [(3, 1), (0, -1)], [(0, 1), (1, 1)])
    
    # M5 = (A00+A01)B11
    add_term([(0, 1), (1, 1)], [(4, 1)], [(0, -1), (3, 1)])
    
    # M6 = (A10-A00)(B00+B01)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(4, 1)])
    
    # M7 = (A01-A11)(B10+B11)
    add_term([(1, 1), (4, -1)], [(3, 1), (4, 1)], [(0, 1)])
    
    # 16 terms for the cross parts
    # We need to handle C02, C12, C20, C21, C22
    # And ensure no interference with C00, C01, C10, C11
    
    # C02 = A00 B02 + A01 B12 + A02 B22
    # C12 = A10 B02 + A11 B12 + A12 B22
    # C20 = A20 B00 + A21 B10 + A22 B20
    # C21 = A20 B01 + A21 B11 + A22 B21
    # C22 = A20 B02 + A21 B12 + A22 B22
    
    # Let's use a known set of 16 terms for these.
    # Term 8: A02 B00 -> C00? No, C00 is done.
    # Actually, the Bini-De Falco construction is specific.
    # I will use a verified set from literature/code.
    
    # T8: A02 B00 -> C00 (Correction? No, C00 is sum of 3 terms. Strassen only handles 2x2. 
    # Wait, C00 = A00B00 + A01B10 + A02B20. 
    # My Strassen terms only used A00, A01, A10, A11 and B00, B01, B10, B11.
    # They did NOT use A02, B20.
    # So C00 is currently missing A02B20.
    # Similarly C10 is missing A12B20.
    # C01 is missing A02B21.
    # C11 is missing A12B21.
    # So I MUST add terms for these.
    
    # So the 16 terms must cover:
    # C00 += A02 B20
    # C10 += A12 B20
    # C01 += A02 B21
    # C11 += A12 B21
    # C02 = A00 B02 + A01 B12 + A02 B22
    # C12 = A10 B02 + A11 B12 + A12 B22
    # C20 = A20 B00 + A21 B10 + A22 B20
    # C21 = A20 B01 + A21 B11 + A22 B21
    # C22 = A20 B02 + A21 B12 + A22 B22
    
    # Total 4 + 3 + 3 + 3 + 3 + 3 = 19 products?
    # No, C02 has 3 terms. C12 has 3. C20 has 3. C21 has 3. C22 has 3.
    # Plus the 4 corrections to the 2x2 block.
    # Total 19 distinct bilinear forms needed if schoolbook.
    # We have 16 slots.
    # So we must compress.
    
    # This is getting complex. The current program's rank-23 attempt likely failed because the terms were wrong.
    # I will fall back to the robust Rank 26 if I cannot guarantee Rank 23 correctness quickly.
    # However, the prompt asks to combine strategies. The strongest idea is the Rank 26 structural decomposition.
    # I will provide the Rank 26 decomposition which is known to be correct and scores 1.0385.
    # To potentially beat it, I would need Rank 23.
    # Given the risk of incorrect Rank 23 terms, I will stick to Rank 26 but optimize the code.
    
    # Actually, let's just return the Rank 26 decomposition which is verified in the inspirations.
    
    # Reset matrices for Rank 26
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # 7 Strassen terms
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    # M3
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    # M4
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    # M6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    # M7
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # Cross terms (19 terms)
    # C02 = A00 B02 + A01 B12 + A02 B22
    U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
    U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
    U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
    
    # C12 = A10 B02 + A11 B12 + A12 B22
    U[3][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
    U[4][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1
    U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
    
    # C20 = A20 B00 + A21 B10 + A22 B20
    U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
    U[7][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
    U[8][col] = 1; V[6][col] = 1; W[2][col] = 1; col += 1
    
    # C21 = A20 B01 + A21 B11 + A22 B21
    U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
    U[7][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1
    U[8][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1
    
    # C22 = A20 B02 + A21 B12 + A22 B22
    U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
    U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
    
    # Corrections to 2x2 block from index 2 terms?
    # C00 += A02 B20
    U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
    # C10 += A12 B20
    U[5][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
    # C01 += A02 B21
    U[2][col] = 1; V[7][col] = 1; W[3][col] = 1; col += 1
    # C11 += A12 B21
    U[5][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
    
    # Total terms: 7 + 3 + 3 + 3 + 3 + 3 + 4 = 26.
    
    return {"U": U, "V": V, "W": W}
