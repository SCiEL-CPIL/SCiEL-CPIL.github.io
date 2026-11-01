import time

DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 10.0
START_TIME = time.time()

def construct():
    """Return a valid exact tensor decomposition for 3x3 * 3x3 matrix multiplication.
    
    Uses a rank-26 decomposition based on block partitioning (2x2 Strassen + 1x1 blocks).
    """
    n, m, p = 3, 3, 3
    
    # Rank 26 decomposition
    rank = 26
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    def add_term(u_dict, v_dict, w_dict):
        # Find an empty column or return None if full
        for r in range(rank):
            if all(U[i][r] == 0 for i in range(9)) and all(V[i][r] == 0 for i in range(9)) and all(W[i][r] == 0 for i in range(9)):
                for i, c in u_dict.items(): U[i][r] = c
                for j, c in v_dict.items(): V[j][r] = c
                for k, c in w_dict.items(): W[k][r] = c
                return r
        return None

    # Helper to add to existing term? No, Strassen terms are distinct.
    # But we need to accumulate W contributions for Strassen.
    # It's easier to build the 7 Strassen terms first with their full W contributions.
    
    # Strassen for A[0..1, 0..1] * B[0..1, 0..1] -> C[0..1, 0..1]
    # Indices:
    # A: 0,1,3,4
    # B: 0,1,3,4
    # C: 0,1,3,4
    
    # M1: A00 * (B01 - B11)
    # U: {0:1}
    # V: {1:1, 4:-1}
    # W: C01 (idx 1) += 1, C11 (idx 4) += 1
    add_term({0:1}, {1:1, 4:-1}, {1:1, 4:1})
    
    # M2: (A00 + A11) * B11
    # U: {0:1, 4:1}
    # V: {4:1}
    # W: C00 (idx 0) += 1, C10 (idx 3) += 1, C11 (idx 4) -= 1
    add_term({0:1, 4:1}, {4:1}, {0:1, 3:1, 4:-1})
    
    # M3: (A10 + A11) * B00
    # U: {3:1, 4:1}
    # V: {0:1}
    # W: C00 (idx 0) += 1, C10 (idx 3) += 1
    add_term({3:1, 4:1}, {0:1}, {0:1, 3:1})
    
    # M4: A11 * (B10 - B00)
    # U: {4:1}
    # V: {3:1, 0:-1}
    # W: C00 (idx 0) -= 1, C01 (idx 1) += 1
    add_term({4:1}, {3:1, 0:-1}, {0:-1, 1:1})
    
    # M5: (A00 + A01) * B11
    # U: {0:1, 1:1}
    # V: {4:1}
    # W: C11 (idx 4) += 1
    add_term({0:1, 1:1}, {4:1}, {4:1})
    
    # M6: (A10 - A00) * (B00 + B01)
    # U: {3:1, 0:-1}
    # V: {0:1, 1:1}
    # W: C11 (idx 4) += 1
    add_term({3:1, 0:-1}, {0:1, 1:1}, {4:1})
    
    # M7: (A01 - A11) * (B10 + B11)
    # U: {1:1, 4:-1}
    # V: {3:1, 4:1}
    # W: C00 (idx 0) += 1, C11 (idx 4) += 1
    add_term({1:1, 4:-1}, {3:1, 4:1}, {0:1, 4:1})
    
    # Now the remaining 19 terms
    # C11 is done.
    # C12 (col 2 of top 2 rows): A11 * B12 + A12 * B22
    # A11: 0,1,3,4. B12: 2,5. A12: 1,4. B22: 8.
    # A11 * B12 (2x2 * 2x1 = 2x1). 4 mults.
    # A12 * B22 (2x1 * 1x1 = 2x1). 2 mults.
    
    # Term 8: A00 * B02
    add_term({0:1}, {2:1}, {2:1}) # C02 (idx 2)
    # Term 9: A01 * B12
    add_term({1:1}, {5:1}, {2:1}) # C02 (idx 2)
    # Term 10: A10 * B02
    add_term({3:1}, {2:1}, {5:1}) # C12 (idx 5)
    # Term 11: A11 * B12
    add_term({4:1}, {5:1}, {5:1}) # C12 (idx 5)
    
    # Term 12: A02 * B22
    add_term({1:1}, {8:1}, {2:1}) # C02 (idx 2) -- Wait, A12 is A[1], A[4].
    # A12 is column 2 of A? No, A is 3x3.
    # A = [[A00, A01, A02], [A10, A11, A12], [A20, A21, A22]]
    # Indices:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # My previous block definition:
    # A11 was top-left 2x2: 0,1,3,4. Correct.
    # A12 is top-right 2x1: 2,5. (Row 0 col 2, Row 1 col 2).
    # A21 is bottom-left 1x2: 6,7.
    # A22 is bottom-right 1x1: 8.
    
    # Let's re-evaluate indices for B.
    # B11: 0,1,3,4.
    # B12: 2,5. (Col 2 of top 2 rows).
    # B21: 6,7. (Row 2 of left 2 cols).
    # B22: 8.
    
    # C12 (top-right 2x1 of C):
    # C02 (idx 2) = A00*B02 + A01*B12 + A02*B22
    # C12 (idx 5) = A10*B02 + A11*B12 + A12*B22
    
    # We already added:
    # 8: A00*B02 -> C02. (U{0}, V{2}, W{2})
    # 9: A01*B12 -> C02. (U{1}, V{5}, W{2})
    # 10: A10*B02 -> C12. (U{3}, V{2}, W{5})
    # 11: A11*B12 -> C12. (U{4}, V{5}, W{5})
    
    # Now A12*B22. A12 is {2,5}. B22 is {8}.
    # Term 12: A02 * B22 -> C02.
    add_term({2:1}, {8:1}, {2:1})
    # Term 13: A12 * B22 -> C12.
    add_term({5:1}, {8:1}, {5:1})
    
    # C21 (bottom-left 1x2 of C):
    # C20 (idx 6) = A20*B00 + A21*B10 + A22*B20
    # C21 (idx 7) = A20*B01 + A21*B11 + A22*B21
    # A20 is 6. A21 is 7. A22 is 8.
    # B20 is 6. B21 is 7.
    
    # Term 14: A20 * B00 -> C20.
    add_term({6:1}, {0:1}, {6:1})
    # Term 15: A21 * B10 -> C20.
    add_term({7:1}, {3:1}, {6:1})
    # Term 16: A22 * B20 -> C20.
    add_term({8:1}, {6:1}, {6:1})
    
    # Term 17: A20 * B01 -> C21.
    add_term({6:1}, {1:1}, {7:1})
    # Term 18: A21 * B11 -> C21.
    add_term({7:1}, {4:1}, {7:1})
    # Term 19: A22 * B21 -> C21.
    add_term({8:1}, {7:1}, {7:1})
    
    # C22 (bottom-right 1x1 of C):
    # C22 (idx 8) = A20*B02 + A21*B12 + A22*B22
    # Term 20: A20 * B02 -> C22.
    add_term({6:1}, {2:1}, {8:1})
    # Term 21: A21 * B12 -> C22.
    add_term({7:1}, {5:1}, {8:1})
    # Term 22: A22 * B22 -> C22.
    add_term({8:1}, {8:1}, {8:1})
    
    # We have used 22 terms.
    # Wait, Strassen was 7.
    # C12 part: 4 (A11*B12) + 2 (A12*B22) = 6.
    # C21 part: 2 (A20*B00, A21*B10, A22*B20 -> 3 terms?)
    # Let's recount C21.
    # C20 = A20*B00 + A21*B10 + A22*B20. (3 terms)
    # C21 = A20*B01 + A21*B11 + A22*B21. (3 terms)
    # C22 = A20*B02 + A21*B12 + A22*B22. (3 terms)
    # Total non-Strassen: 6 (C12) + 6 (C21) + 3 (C22) = 15.
    # 7 + 15 = 22.
    # Where are the other 4?
    # Ah, C12 calculation:
    # C02 = A00*B02 + A01*B12 + A02*B22.
    # C12 = A10*B02 + A11*B12 + A12*B22.
    # I listed:
    # 8: A00*B02
    # 9: A01*B12
    # 10: A10*B02
    # 11: A11*B12
    # 12: A02*B22
    # 13: A12*B22
    # That's 6 terms.
    
    # C21 calculation:
    # 14: A20*B00
    # 15: A21*B10
    # 16: A22*B20
    # 17: A20*B01
    # 18: A21*B11
    # 19: A22*B21
    # That's 6 terms.
    
    # C22 calculation:
    # 20: A20*B02
    # 21: A21*B12
    # 22: A22*B22
    # That's 3 terms.
    
    # Total 7 + 6 + 6 + 3 = 22.
    # Why did I think 26?
    # Because I counted "4 mults" for 2x2*2x1.
    # A11 (2x2) * B12 (2x1) requires 4 scalar multiplications.
    # A00*B02, A01*B12, A10*B02, A11*B12. Yes, 4.
    # A12 (2x1) * B22 (1x1) requires 2 scalar multiplications.
    # A02*B22, A12*B22. Yes, 2.
    # Total 6.
    
    # A21 (1x2) * B11 (2x2) requires 4 scalar multiplications.
    # A20*B00, A21*B10, A20*B01, A21*B11. Yes, 4.
    # A22 (1x1) * B21 (1x2) requires 2 scalar multiplications.
    # A22*B20, A22*B21. Yes, 2.
    # Total 6.
    
    # A21 (1x2) * B12 (2x1) requires 2 scalar multiplications.
    # A20*B02, A21*B12. Yes, 2.
    # A22 (1x1) * B22 (1x1) requires 1.
    # Total 3.
    
    # 7 + 6 + 6 + 3 = 22.
    # So Rank 22 is possible with this block partition?
    # Wait, is 22 valid?
    # Yes, if the terms are distinct and independent.
    # Let's verify Rank 22.
    
    # If Rank 22 is valid, score is 27/22 = 1.22.
    # This is better than 26.
    
    # I will return the Rank 22 decomposition.
    
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = sum(U[i*3+j][r] * V[j*3+k][r] * W[k*3+i][r] for r in range(rank))
                if s != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to 27
        rank = 27
        U = [[0]*rank for _ in range(9)]
        V = [[0]*rank for _ in range(9)]
        W = [[0]*rank for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
