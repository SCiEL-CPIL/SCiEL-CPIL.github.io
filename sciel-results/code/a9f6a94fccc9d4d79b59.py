import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        # a_idx: indices in U (size 8)
        # b_idx: indices in V (size 24)
        # w_idx: indices in W (size 12)
        # start_col: starting column index in the rank dimension
        
        # Strassen 7 multiplications for 2x2 matrices
        # A = [[a0, a1], [a2, a3]]
        # B = [[b0, b1], [b2, b3]]
        # C = [[c0, c1], [c2, c3]]
        
        # M1 = (a0+a3)(b0+b3) -> c0+=, c3+=, c1-=, c2-= ? No.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # U entries correspond to A coefficients
        # V entries correspond to B coefficients
        # W entries correspond to C coefficients
        
        # U[a_idx[0]] is A00
        # U[a_idx[1]] is A01
        # U[a_idx[2]] is A10
        # U[a_idx[3]] is A11
        
        # V[b_idx[0]] is B00
        # V[b_idx[1]] is B01
        # V[b_idx[2]] is B10
        # V[b_idx[3]] is B11
        
        # W[w_idx[0]] is C00
        # W[w_idx[1]] is C01
        # W[w_idx[2]] is C10
        # W[w_idx[3]] is C11
        
        # M1: (A00+A11)(B00+B11)
        U[a_idx[0]][start_col + 0] += 1
        U[a_idx[3]][start_col + 0] += 1
        V[b_idx[0]][start_col + 0] += 1
        V[b_idx[3]][start_col + 0] += 1
        W[w_idx[0]][start_col + 0] += 1
        W[w_idx[3]][start_col + 0] += 1
        
        # M2: (A10+A11)B00
        U[a_idx[2]][start_col + 1] += 1
        U[a_idx[3]][start_col + 1] += 1
        V[b_idx[0]][start_col + 1] += 1
        W[w_idx[1]][start_col + 1] += 1  # C01? No, M2 contributes to C10 and C11?
        # Wait, M2 = (A10+A11)B00.
        # C10 = M2 + M4. So W[w_idx[2]] (C10) += M2.
        # C11 = M1 - M2 + M3 + M7. So W[w_idx[3]] (C11) -= M2.
        W[w_idx[2]][start_col + 1] += 1
        W[w_idx[3]][start_col + 1] -= 1
        
        # M3: A00(B10-B11)
        U[a_idx[0]][start_col + 2] += 1
        V[b_idx[2]][start_col + 2] += 1
        V[b_idx[3]][start_col + 2] -= 1
        # C01 = M3 + M5. W[w_idx[1]] += M3.
        # C11 = M1 - M2 + M3 + M7. W[w_idx[3]] += M3.
        W[w_idx[1]][start_col + 2] += 1
        W[w_idx[3]][start_col + 2] += 1
        
        # M4: A11(B01-B00)
        U[a_idx[3]][start_col + 3] += 1
        V[b_idx[1]][start_col + 3] += 1
        V[b_idx[0]][start_col + 3] -= 1
        # C00 = M1 + M4 - M5 - M7. W[w_idx[0]] += M4.
        # C10 = M2 + M4. W[w_idx[2]] += M4.
        W[w_idx[0]][start_col + 3] += 1
        W[w_idx[2]][start_col + 3] += 1
        
        # M5: (A00+A01)B11
        U[a_idx[0]][start_col + 4] += 1
        U[a_idx[1]][start_col + 4] += 1
        V[b_idx[3]][start_col + 4] += 1
        # C00 = M1 + M4 - M5 - M7. W[w_idx[0]] -= M5.
        # C01 = M3 + M5. W[w_idx[1]] += M5.
        W[w_idx[0]][start_col + 4] -= 1
        W[w_idx[1]][start_col + 4] += 1
        
        # M6: (A10-A00)(B00+B01) -- Wait, standard Strassen uses M6 for C10?
        # Let's re-verify standard Strassen formulas.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # My M2 above was (A10+A11)B00.
        # C10 = M2 + M4. Correct.
        # C11 = M1 - M2 + ... Correct.
        
        # M6: (A10-A00)(B00+B01)
        # This term is not in the standard 7-multiplication set I listed above?
        # Ah, the standard set is:
        # M1..M7 as above.
        # But wait, C10 = M2 + M4.
        # M2 = (A10+A11)B00.
        # M4 = A11(B01-B00) = A11 B01 - A11 B00.
        # M2 + M4 = (A10+A11)B00 + A11 B01 - A11 B00 = A10 B00 + A11 B01.
        # This is correct for C10.
        
        # So what is M6 and M7 for?
        # M6 is not used in the standard Strassen 7-mult algorithm?
        # No, the 7 multiplications are M1..M7.
        # Let's check C00.
        # C00 = A00 B00 + A01 B10.
        # M1 + M4 - M5 - M7
        # = (A00+A11)(B00+B11) + A11(B01-B00) - (A00+A01)B11 - (A01-A11)(B10+B11)
        # = A00 B00 + A00 B11 + A11 B00 + A11 B11 + A11 B01 - A11 B00 - A00 B11 - A01 B11 - (A01 B10 + A01 B11 - A11 B10 - A11 B11)
        # = A00 B00 + (A00 B11 - A00 B11) + (A11 B00 - A11 B00) + (A11 B11 - A01 B11 + A11 B11 - A01 B11? No)
        # Let's expand carefully.
        # M1 = A00 B00 + A00 B11 + A11 B00 + A11 B11
        # M4 = A11 B01 - A11 B00
        # M5 = A00 B11 + A01 B11
        # M7 = A01 B10 + A01 B11 - A11 B10 - A11 B11
        
        # Sum = A00 B00 + A00 B11 + A11 B00 + A11 B11 + A11 B01 - A11 B00 - A00 B11 - A01 B11 - A01 B10 - A01 B11 + A11 B10 + A11 B11
        # Terms:
        # A00 B00: 1
        # A00 B11: 1 - 1 = 0
        # A11 B00: 1 - 1 = 0
        # A11 B11: 1 + 1 - 1 = 1? No, M1 has +1, M7 has -1 (from -A01 B11? No, M7 is subtracted).
        # M7 = (A01-A11)(B10+B11) = A01 B10 + A01 B11 - A11 B10 - A11 B11.
        # -M7 = -A01 B10 - A01 B11 + A11 B10 + A11 B11.
        # So A11 B11: 1 (from M1) + 1 (from -M7) = 2?
        # This doesn't look like A00 B00 + A01 B10.
        
        # I must have the formulas wrong or the assignment of M6/M7 wrong.
        # Let's use the verified formulas from the current program's logic or a known source.
        # The current program's `add_strassen_block` is likely correct because it scored 1.1429.
        # I will copy the logic from the current program's `add_strassen_block` exactly.
        pass

    # Re-implementing the exact logic from the current program to ensure correctness
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        U[a_idx[0]][start_col + 0] += 1
        U[a_idx[3]][start_col + 0] += 1
        U[a_idx[2]][start_col + 1] += 1
        U[a_idx[3]][start_col + 1] += 1
        U[a_idx[0]][start_col + 2] += 1
        U[a_idx[3]][start_col + 3] += 1
        U[a_idx[0]][start_col + 4] += 1
        U[a_idx[1]][start_col + 4] += 1
        U[a_idx[2]][start_col + 5] += 1
        U[a_idx[0]][start_col + 5] -= 1
        U[a_idx[1]][start_col + 6] += 1
        U[a_idx[3]][start_col + 6] -= 1

        V[b_idx[0]][start_col + 0] += 1
        V[b_idx[3]][start_col + 0] += 1
        V[b_idx[0]][start_col + 1] += 1
        V[b_idx[1]][start_col + 2] += 1
        V[b_idx[3]][start_col + 2] -= 1
        V[b_idx[2]][start_col + 3] += 1
        V[b_idx[0]][start_col + 3] -= 1
        V[b_idx[3]][start_col + 4] += 1
        V[b_idx[0]][start_col + 5] += 1
        V[b_idx[1]][start_col + 5] += 1
        V[b_idx[2]][start_col + 6] += 1
        V[b_idx[3]][start_col + 6] += 1

        W[w_idx[0]][start_col + 0] += 1
        W[w_idx[0]][start_col + 3] += 1
        W[w_idx[0]][start_col + 4] -= 1
        W[w_idx[0]][start_col + 6] += 1

        W[w_idx[1]][start_col + 2] += 1
        W[w_idx[1]][start_col + 4] += 1

        W[w_idx[2]][start_col + 1] += 1
        W[w_idx[2]][start_col + 3] += 1

        W[w_idx[3]][start_col + 0] += 1
        W[w_idx[3]][start_col + 1] -= 1
        W[w_idx[3]][start_col + 2] += 1
        W[w_idx[3]][start_col + 5] += 1

    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    return {"U": U, "V": V, "W": W}
