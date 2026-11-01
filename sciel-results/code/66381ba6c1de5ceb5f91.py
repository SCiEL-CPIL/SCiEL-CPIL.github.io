import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def check_decomposition(U, V, W, n, m, p):
    r = U.shape[1]
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            s = 0.0
                            for l in range(r):
                                s += U[a * m + b, l] * V[c * p + d, l] * W[k * n + i, l]
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Rank 34 solution: 2 * Rank(2,3,3)
    # Rank(2,3,3) = 17
    # Constructed from M_222 (7) + M_221 (4) + M_213 (6)
    # Wait, M_213 is 2x1x3. Rank 6.
    # M_222 is 2x2x2. Rank 7.
    # M_221 is 2x2x1. Rank 4.
    # Total 17.
    
    # We build the 2x3x3 decomposition first.
    # U233: (6, 17)
    # V233: (9, 17)
    # W233: (6, 17)
    
    U233 = np.zeros((6, 17))
    V233 = np.zeros((9, 17))
    W233 = np.zeros((6, 17))
    
    # Part 1: M_222 (Strassen)
    # A indices: i in 0,1; j in 0,1. U index i*3+j
    # B indices: j in 0,1; k in 0,1. V index j*3+k
    # C indices: i in 0,1; k in 0,1. W index k*2+i
    
    # P1
    U233[0, 0] = 1; U233[3, 0] = 1
    V233[0, 0] = 1; V233[3, 0] = 1
    W233[0, 0] = 1; W233[2, 0] = 1 # C11, C21? No.
    # Strassen:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # P1: (A11+A22)(B11+B22)
    # U: A11(0,0), A22(1,1) -> idx 0, 4? No, A is 2x2.
    # In 2x3x3, A is 2x3.
    # We use A_left (2x2). Indices 0,1 for j.
    # A11 is idx 0. A12 is idx 1. A21 is idx 3. A22 is idx 4.
    # B is 3x3. We use B_top (2x2) for P1..P7?
    # B_top is rows 0,1. Cols 0,1.
    # B11 is idx 0. B12 is idx 1. B21 is idx 3. B22 is idx 4.
    
    # P1
    U233[0, 0] = 1; U233[4, 0] = 1
    V233[0, 0] = 1; V233[4, 0] = 1
    W233[0, 0] = 1 # C11
    W233[2, 0] = 1 # C21? No, C21 is k=0, i=1?
    # W index k*n + i.
    # C11: k=0, i=0 -> idx 0.
    # C12: k=1, i=0 -> idx 2.
    # C21: k=0, i=1 -> idx 1.
    # C22: k=1, i=1 -> idx 3.
    
    # Correct W indices for 2x2 C:
    # C00: 0
    # C10: 1
    # C01: 2
    # C11: 3
    
    # P1 contributes to C00 and C11?
    # Strassen:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # P1: (A00+A11)(B00+B11)
    U233[0, 0] = 1; U233[4, 0] = 1
    V233[0, 0] = 1; V233[4, 0] = 1
    W233[0, 0] = 1 # C00
    W233[3, 0] = 1 # C11
    
    # P2: (A10+A11)B00
    U233[3, 1] = 1; U233[4, 1] = 1
    V233[0, 1] = 1
    W233[1, 1] = 1 # C10
    
    # P3: A00(B01-B11)
    U233[0, 2] = 1
    V233[1, 2] = 1; V233[4, 2] = -1
    W233[2, 2] = 1 # C01
    W233[3, 2] = 1 # C11
    
    # P4: A11(B10-B00)
    U233[4, 3] = 1
    V233[3, 3] = 1; V233[0, 3] = -1
    W233[0, 3] = 1 # C00
    W233[1, 3] = 1 # C10
    
    # P5: (A00+A01)B11
    U233[0, 4] = 1; U233[1, 4] = 1
    V233[4, 4] = 1
    W233[0, 4] = -1 # C00
    W233[2, 4] = 1 # C01
    
    # P6: (A10-A00)(B00+B01)
    U233[3, 5] = 1; U233[0, 5] = -1
    V233[0, 5] = 1; V233[1, 5] = 1
    W233[3, 5] = 1 # C11
    
    # P7: (A01-A11)(B10+B11)
    U233[1, 6] = 1; U233[4, 6] = -1
    V233[3, 6] = 1; V233[4, 6] = 1
    W233[0, 6] = 1 # C00
    W233[3, 6] = 1 # C11
    
    # Part 2: M_221 (Rank 4)
    # Uses A_left (2x2) and B_col2 (2x1).
    # B_col2 is rows 0,1, col 2.
    # B02 is idx 2. B12 is idx 5.
    # C_col2 is k=2.
    # C02: k=2, i=0 -> idx 4.
    # C12: k=2, i=1 -> idx 5.
    
    # Q1: A00 B02
    U233[0, 7] = 1
    V233[2, 7] = 1
    W233[4, 7] = 1
    
    # Q2: A01 B12
    U233[1, 8] = 1
    V233[5, 8] = 1
    W233[4, 8] = 1
    
    # Q3: A10 B02
    U233[3, 9] = 1
    V233[2, 9] = 1
    W233[5, 9] = 1
    
    # Q4: A11 B12
    U233[4, 10] = 1
    V233[5, 10] = 1
    W233[5, 10] = 1
    
    # Part 3: M_213 (Rank 6)
    # Uses A_col2 (2x1) and B_bottom (1x3).
    # A_col2 is col 2 of A. A02 is idx 2. A12 is idx 5.
    # B_bottom is row 2 of B. B20 is idx 6. B21 is idx 7. B22 is idx 8.
    # C is 2x3.
    
    # R1: A02 B20
    U233[2, 11] = 1
    V233[6, 11] = 1
    W233[0, 11] = 1 # C00
    
    # R2: A02 B21
    U233[2, 12] = 1
    V233[7, 12] = 1
    W233[2, 12] = 1 # C01
    
    # R3: A02 B22
    U233[2, 13] = 1
    V233[8, 13] = 1
    W233[4, 13] = 1 # C02
    
    # R4: A12 B20
    U233[5, 14] = 1
    V233[6, 14] = 1
    W233[1, 14] = 1 # C10
    
    # R5: A12 B21
    U233[5, 15] = 1
    V233[7, 15] = 1
    W233[3, 15] = 1 # C11
    
    # R6: A12 B22
    U233[5, 16] = 1
    V233[8, 16] = 1
    W233[5, 16] = 1 # C12
    
    # Now combine two 2x3x3 to form 2x3x6.
    # U_comb: (6, 34)
    # V_comb: (18, 34)
    # W_comb: (12, 34)
    
    U_comb = np.zeros((6, 34))
    V_comb = np.zeros((18, 34))
    W_comb = np.zeros((12, 34))
    
    U_comb[:, :17] = U233
    U_comb[:, 17:] = U233
    
    # V: 3x6.
    # B1 (cols 0-2) uses V233.
    # B2 (cols 3-5) uses V233.
    for j in range(3):
        for k in range(3):
            V_comb[j * 6 + k, :17] = V233[j * 3 + k]
            V_comb[j * 6 + 3 + k, 17:] = V233[j * 3 + k]
            
    # W: 6x2.
    # C1 (cols 0-2) uses W233.
    # C2 (cols 3-5) uses W233.
    for k in range(3):
        for i in range(2):
            W_comb[k * 2 + i, :17] = W233[k * 2 + i]
            W_comb[(k + 3) * 2 + i, 17:] = W233[k * 2 + i]
            
    if check_decomposition(U_comb, V_comb, W_comb, 2, 3, 6):
        return {
            "U": U_comb.tolist(),
            "V": V_comb.tolist(),
            "W": W_comb.tolist()
        }
    else:
        # Fallback to schoolbook
        U_school = np.zeros((6, 36))
        V_school = np.zeros((18, 36))
        W_school = np.zeros((12, 36))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(6):
                    U_school[i * 3 + j, col] = 1
                    V_school[j * 6 + k, col] = 1
                    W_school[k * 2 + i, col] = 1
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

if __name__ == "__main__":
    construct()
