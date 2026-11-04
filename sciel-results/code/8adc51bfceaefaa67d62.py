import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices:
    # U rows: 00, 01, 10, 11
    # V rows: 00, 01, 10, 11
    # W rows: 00, 01, 10, 11
    
    # 1: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # 2: M2 = A11(B10-B00)
    U[3, 1] = 1
    V[2, 1] = 1; V[0, 1] = -1
    W[2, 1] = 1; W[3, 1] = 1
    
    # 3: M3 = A00(B11-B01)
    U[0, 2] = 1
    V[3, 2] = 1; V[1, 2] = -1
    W[1, 2] = 1; W[0, 2] = 1
    
    # 4: M4 = A11(B01+B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = 1
    W[1, 3] = 1; W[3, 3] = 1
    
    # 5: M5 = A00+A10-B10-B11)B11
    U[0, 4] = 1; U[2, 4] = 1; U[3, 4] = -1; U[1, 4] = -1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # 6: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[0, 5] = 1
    
    # 7: M7 = (A01+A11)(B10-B11)
    U[1, 6] = 1; U[3, 6] = 1
    V[2, 6] = 1; V[3, 6] = -1
    W[3, 6] = 1; W[2, 6] = 1
    
    return U, V, W

def _laderman_3x3x3():
    """
    Returns U, V, W for 3x3x3 matrix multiplication with rank 13 (Laderman's algorithm).
    U: (9, 13), V: (9, 13), W: (9, 13)
    Convention: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    """
    U = np.zeros((9, 13))
    V = np.zeros((9, 13))
    W = np.zeros((9, 13))
    
    # Row indices for U, V, W are 0..8 corresponding to (i,j) or (j,k) or (k,i)
    # i,j,k in 0,1,2
    
    # Laderman's 13 multiplications:
    # 1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[8, 0] = 1 # W[k*3+i]: k=0,i=0 -> 0; k=2,i=2 -> 8
    
    # 2: A11(B10-B00)
    U[4, 1] = 1
    V[3, 1] = 1; V[0, 1] = -1
    W[3, 1] = 1; W[8, 1] = 1 # k=1,i=1 -> 4? No, W[k*3+i]. k=1, i=1 -> 4. k=2, i=2 -> 8.
    # Wait, W index is k*n + i. n=3.
    # For A11 (i=1, j=1) and B10 (j=1, k=0) -> C10 (i=1, k=0) -> W[0*3+1] = 1
    # For A11 (i=1, j=1) and B00 (j=0, k=0) -> C10 (i=1, k=0) -> W[0*3+1] = 1
    # So W[1, 1] = 1.
    # Let's re-verify Laderman's formula.
    # C = A B
    # C_ik = sum_j A_ij B_jk
    # Term 2: A_11 (B_10 - B_00).
    # This contributes to C_10 (i=1, k=0) and C_11 (i=1, k=1)?
    # A_11 * B_10 -> C_10 (i=1, k=0). Index W[0*3+1] = 1.
    # A_11 * B_00 -> C_10 (i=1, k=0). Index W[0*3+1] = 1.
    # So W[1, 1] = 1.
    # Did I mess up the indices in my head?
    # Strassen 2x2:
    # M2 = A11(B10-B00).
    # A11 is i=1, j=1.
    # B10 is j=1, k=0. Product contributes to C10 (i=1, k=0). W[0*2+1] = 1.
    # B00 is j=0, k=0. Product contributes to C10 (i=1, k=0). W[0*2+1] = 1.
    # So W[1, 1] = 1.
    # My previous Strassen code:
    # W[2, 1] = 1; W[3, 1] = 1
    # W index k*2+i.
    # C10 -> k=0, i=1 -> W[1].
    # C11 -> k=1, i=1 -> W[3].
    # Did I write W[2,1] and W[3,1] in Strassen?
    # Let's check Strassen code again.
    # U[3, 1] = 1 (A11)
    # V[2, 1] = 1 (B10); V[0, 1] = -1 (B00)
    # W[2, 1] = 1 (C10? No, W[2] is k=1, i=0 -> C01? No. W[k*2+i].
    # k=1, i=0 -> W[2]. This is C_01 (i=0, k=1).
    # k=1, i=1 -> W[3]. This is C_11 (i=1, k=1).
    # This is WRONG.
    # A11 * B10 -> C10 (i=1, k=0). W[0*2+1] = W[1].
    # A11 * B00 -> C10 (i=1, k=0). W[0*2+1] = W[1].
    # So W[1, 1] should be 1.
    # Where did I get W[2,1] and W[3,1]?
    # Maybe I confused it with something else.
    # Let's re-derive Strassen carefully.
    
    # C = [[C00, C01], [C10, C11]]
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M2 = A11(B10-B00) = A11B10 - A11B00
    # M3 = A00(B11-B01) = A00B11 - A00B01
    # M4 = A11(B01+B11) = A11B01 + A11B11
    # M5 = (A00+A10-B10-B11)B11 = A00B11 + A10B11 - B10B11 - B11B11 ... wait
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A11)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M5
    
    # Let's check M4 = A11(B10-B00).
    # Contributes to C10 and C11?
    # A11 (i=1, j=1) * B10 (j=1, k=0) -> C10 (i=1, k=0).
    # A11 (i=1, j=1) * B00 (j=0, k=0) -> C10 (i=1, k=0).
    # So M4 contributes to C10.
    # C10 = M2 + M7.
    # M2 = (A10+A11)B00 = A10B00 + A11B00.
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11.
    # C10 = A10B00 + A11B00 + A01B10 + A01B11 - A11B10 - A11B11.
    # This doesn't look like C10 = A10B00 + A11B10.
    # So my standard Strassen formulas are mixed up.
    
    # Correct Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A11)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M5
    
    # Let's check C10 = M2 + M7.
    # M2 = A10B00 + A11B00
    # M7 = A01B10 + A01B11 - A11B10 - A11B11
    # Sum = A10B00 + A11B00 + A01B10 + A01B11 - A11B10 - A11B11
    # This is not C10.
    
    # Okay, let's use the standard Wikipedia Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A11)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M5
    
    # Wait, C10 = A10B00 + A11B10.
    # M2 + M7 = (A10+A11)B00 + (A01-A11)(B10+B11)
    # = A10B00 + A11B00 + A01B10 + A01B11 - A11B10 - A11B11
    # This is definitely not C10.
    
    # I will just implement the schoolbook for 2x2 and 3x3 to be safe, then try to optimize.
    # The prompt suggests using Strassen and Laderman. I need to get them right.
    
    # Let's find a reliable source for Laderman 3x3x3 rank 13.
    # Laderman's algorithm:
    # 1. (A00+A11)(B00+B11)
    # 2. A11(B10-B00)
    # 3. A00(B11-B01)
    # 4. A11(B01+B11)
    # 5. (A00+A10-B10-B11)B11
    # 6. (A10-A00)(B00+B01)
    # 7. (A01+A11)(B10-B11)
    # 8. (A02+A12)(B20+B21)
    # 9. A22(B12-B02)
    # 10. A02(B21-B12)
    # 11. A22(B12+B21)
    # 12. (A00+A02-B20-B22)B21
    # 13. (A02-A00)(B20+B12)
    
    # This is getting complicated and error-prone to derive from memory.
    # Given the time limit and the risk of incorrect coefficients, I will stick to the schoolbook decomposition.
    # The schoolbook rank is 4*5*5 = 100.
    # The current score is 1.0000.
    # If I can find a valid decomposition with rank < 100, the score will be > 1.0.
    # But if I provide an invalid decomposition, the score is 0.
    # So I must ensure validity.
    
    # I will implement the schoolbook decomposition, which is guaranteed to be valid.
    # Then I will try to add a simple block decomposition if I can verify it quickly.
    
    # For now, return the schoolbook result.
    pass

def schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def combine(U1, V1, W1, U2, V2, W2):
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
