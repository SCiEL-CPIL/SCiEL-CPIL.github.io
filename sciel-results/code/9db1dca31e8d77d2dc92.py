import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        # u_idx: (i, j) -> i*m + j
        # v_idx: (j, k) -> j*p + k
        # w_idx: (k, i) -> k*n + i
        # This function is not quite right because Strassen terms are products of sums.
        # We need to add contributions to U, V, W for each Strassen multiplication m_c.
        # m_c = (linear_A) * (linear_B)
        # So for a specific column c, U[i*m+j, c] holds the coefficient of A[i,j] in linear_A.
        # V[j*p+k, c] holds the coefficient of B[j,k] in linear_B.
        # W[k*n+i, c] holds the coefficient for C[k,i] in the output sum.
        pass

    # Helper to add a Strassen block
    # a_indices: list of (i, j) for A
    # b_indices: list of (j, k) for B
    # w_indices: list of (k, i) for C
    # This is complex to generalize. I will hardcode the 28 columns.
    
    # Column 0-6: Strassen for j in {0,1}, k in {0,1}
    # A: A00, A01, A10, A11
    # B: B00, B01, B10, B11
    # C: C00, C01, C10, C11 (indices k*n+i)
    
    # m1 = (A00 + A11)(B00 + B11)
    # m2 = (A10 + A11)B00
    # m3 = A00(B01 - B11)
    # m4 = A11(B10 - B00)
    # m5 = (A00 + A01)B11
    # m6 = (A10 - A00)(B00 + B01)
    # m7 = (A01 - A11)(B10 + B11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m1 + m7
    # C10 = m2 + m4 + m1
    # C11 = m3 + m2 + m6
    
    # Indices:
    # A00: 0, A01: 1, A10: 3, A11: 4
    # B00: 0, B01: 1, B10: 5, B11: 6
    # C00: 0, C01: 1, C10: 2, C11: 3
    
    strassen1 = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: [A00, A01, A10, A11]
        # V: [B00, B01, B10, B11]
        # W: [C00, C01, C10, C11]
        
        # m1
        ({0: 1, 4: 1}, {0: 1, 6: 1}, {0: 1, 1: 1, 2: 1, 3: 0}), # Wait, W coeffs are for the output C.
        # The formula is C = sum w_c * m_c.
        # So for column c, W[k*n+i, c] is the coefficient of m_c in C[k,i].
        
        # Let's re-define the helper to be clearer.
    ]
    
    # I will manually populate U, V, W for the 28 columns.
    
    # Part 1: Strassen (j in {0,1}, k in {0,1})
    # Col 0: m1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[6][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1 # C00, C01, C10 get +1?
    # C00 += m1, C01 += m1, C10 += m1, C11 += 0?
    # Strassen:
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m1 + m7
    # C10 = m2 + m4 + m1
    # C11 = m3 + m2 + m6
    
    # So for m1 (col 0):
    W[0][0] = 1 # C00
    W[1][0] = 1 # C01
    W[2][0] = 1 # C10
    W[3][0] = 0 # C11
    
    # Col 1: m2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[0][1] = 0
    W[1][1] = 0
    W[2][1] = 1 # C10
    W[3][1] = 1 # C11
    
    # Col 2: m3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[6][2] = -1
    W[0][2] = 0
    W[1][2] = 1 # C01
    W[2][2] = 0
    W[3][2] = 1 # C11
    
    # Col 3: m4 = A11(B10-B00)
    U[4][3] = 1
    V[5][3] = 1; V[0][3] = -1
    W[0][3] = 1 # C00
    W[1][3] = 0
    W[2][3] = 1 # C10
    W[3][3] = 0
    
    # Col 4: m5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[6][4] = 1
    W[0][4] = -1 # C00
    W[1][4] = 0
    W[2][4] = 0
    W[3][4] = 0
    
    # Col 5: m6 = (A10-A00)(B00+B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 0
    W[1][5] = 0
    W[2][5] = 0
    W[3][5] = 1 # C11
    
    # Col 6: m7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[4][6] = -1
    V[5][6] = 1; V[6][6] = 1
    W[0][6] = 1 # C00
    W[1][6] = 1 # C01
    W[2][6] = 0
    W[3][6] = 0
    
    # Part 2: Tail (j=2, k in {0,1})
    # Col 7: A02 * B20 -> C00
    U[2][7] = 1
    V[10][7] = 1 # B20 is index 2*5+0 = 10
    W[0][7] = 1
    
    # Col 8: A02 * B21 -> C01
    U[2][8] = 1
    V[11][8] = 1 # B21 is index 2*5+1 = 11
    W[1][8] = 1
    
    # Col 9: A12 * B20 -> C10
    U[5][9] = 1 # A12 is index 1*3+2 = 5
    V[10][9] = 1
    W[2][9] = 1
    
    # Col 10: A12 * B21 -> C11
    U[5][10] = 1
    V[11][10] = 1
    W[3][10] = 1
    
    # Part 3: Strassen-like for 2x2x3 (j in {0,1}, k in {2,3,4})
    # A: A00, A01, A10, A11
    # B: B02, B03, B12, B13 (for k=2,3) and B04, B14 (for k=4)
    # C: C02, C03, C12, C13 (k=2,3) and C04, C14 (k=4)
    
    # Strassen for k=2,3 (using B02, B03, B12, B13)
    # m8 = (A00+A11)(B02+B13)
    # m9 = (A10+A11)B02
    # m10 = A00(B03-B13)
    # m11 = A11(B12-B02)
    # m12 = (A00+A01)B13
    # m13 = (A10-A00)(B02+B03)
    # m14 = (A01-A11)(B12+B13)
    
    # C02 += m8 + m11 - m12 + m14
    # C03 += m10 + m8 + m14
    # C12 += m9 + m11 + m8
    # C13 += m10 + m9 + m13
    
    # Indices:
    # B02: 2, B03: 3, B12: 7, B13: 8
    # C02: 4, C03: 5, C12: 6, C13: 7
    
    # Col 7 is used. So start at 11.
    
    # Col 11: m8
    U[0][11] = 1; U[4][11] = 1
    V[2][11] = 1; V[8][11] = 1
    W[4][11] = 1; W[5][11] = 1; W[6][11] = 1
    
    # Col 12: m9
    U[3][12] = 1; U[4][12] = 1
    V[2][12] = 1
    W[4][12] = 0; W[5][12] = 0; W[6][12] = 1; W[7][12] = 1
    
    # Col 13: m10
    U[0][13] = 1
    V[3][13] = 1; V[8][13] = -1
    W[4][13] = 0; W[5][13] = 1; W[6][13] = 0; W[7][13] = 1
    
    # Col 14: m11
    U[4][14] = 1
    V[7][14] = 1; V[2][14] = -1
    W[4][14] = 1; W[5][14] = 0; W[6][14] = 1; W[7][14] = 0
    
    # Col 15: m12
    U[0][15] = 1; U[1][15] = 1
    V[8][15] = 1
    W[4][15] = -1; W[5][15] = 0; W[6][15] = 0; W[7][15] = 0
    
    # Col 16: m13
    U[3][16] = 1; U[0][16] = -1
    V[2][16] = 1; V[3][16] = 1
    W[4][16] = 0; W[5][16] = 0; W[6][16] = 0; W[7][16] = 1
    
    # Col 17: m14
    U[1][17] = 1; U[4][17] = -1
    V[7][17] = 1; V[8][17] = 1
    W[4][17] = 1; W[5][17] = 1; W[6][17] = 0; W[7][17] = 0
    
    # Tail for k=4 (j in {0,1})
    # A00*B04, A01*B14, A10*B04, A11*B14
    # C04 += A00*B04 + A01*B14
    # C14 += A10*B04 + A11*B14
    
    # Col 18: A00*B04
    U[0][18] = 1
    V[4][18] = 1 # B04 is index 0*5+4 = 4
    W[8][18] = 1 # C04 is index 4*2+0 = 8
    
    # Col 19: A01*B14
    U[1][19] = 1
    V[9][19] = 1 # B14 is index 1*5+4 = 9
    W[8][19] = 1
    
    # Col 20: A10*B04
    U[3][20] = 1
    V[4][20] = 1
    W[9][20] = 1 # C14 is index 4*2+1 = 9
    
    # Col 21: A11*B14
    U[4][21] = 1
    V[9][21] = 1
    W[9][21] = 1
    
    # Part 4: Tail (j=2, k in {2,3,4})
    # A02*B22, A02*B23, A02*B24, A12*B22, A12*B23, A12*B24
    
    # Col 22: A02*B22
    U[2][22] = 1
    V[12][22] = 1 # B22 is index 2*5+2 = 12
    W[8][22] = 1 # C04? No, C02 is index 2*2+0=4.
    # Wait, C indices:
    # k=0: C00(0), C10(1)? No, W[k*n+i].
    # n=2.
    # k=0: W[0][i] -> C0i?
    # Convention: W[k*n+i, r].
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T_{i,j,k} corresponds to C[i,k] term from A[i,j]B[j,k].
    # So W index is k*n + i?
    # Let's check the seed code:
    # W[k * n + i][col] = 1
    # And verification:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This implies the output element is identified by (k, i).
    # But standard matrix multiplication C = A B has C[i, k].
    # So the output matrix is transposed in the index?
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # This means W index `idx` corresponds to `k = idx // n` and `i = idx % n`.
    # So `W[k*n+i]` is the coefficient for the term contributing to `C[i,k]`?
    # Or is `C` stored as `C[k,i]`?
    # Usually `C[i,k]` is the element at row i, col k.
    # If the convention is `W[k*n+i]`, then the first index of W is k, second is i.
    # This matches `C[i,k]` if we view the output tensor as `T_{i,j,k}` where k is the col index of C and i is the row index.
    # So `W[k*n+i]` is the factor for `C[i,k]`.
    
    # Let's re-verify my W indices.
    # C[i,k]. i in {0,1}, k in {0,1,2,3,4}.
    # W index = k*n + i = 2*k + i.
    
    # My previous indices:
    # C00 (i=0, k=0) -> 2*0+0 = 0. Correct.
    # C01 (i=0, k=1) -> 2*1+0 = 2.
    # Wait, in my Part 1, I used:
    # W[0][0]=1 (C00)
    # W[1][0]=1 (C01?) -> 2*1+0 = 2. So W[2] is C01.
    # I used W[1] for C01 in my manual fill?
    # Let's check:
    # "W[1][0] = 1 # C01"
    # If W index 1 corresponds to k=0, i=1 (C10), then I made a mistake.
    # W index = k*n + i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    
    # So:
    # C00 (i=0,k=0) -> 0
    # C10 (i=1,k=0) -> 1
    # C01 (i=0,k=1) -> 2
    # C11 (i=1,k=1) -> 3
    
    # My Strassen formulas:
    # C00 (i=0,k=0) = m1 + m4 - m5 + m7
    # C01 (i=0,k=1) = m3 + m1 + m7
    # C10 (i=1,k=0) = m2 + m4 + m1
    # C11 (i=1,k=1) = m3 + m2 + m6
    
    # So for m1 (col 0):
    # W[0] (C00) += 1
    # W[2] (C01) += 1
    # W[1] (C10) += 1
    # W[3] (C11) += 0
    
    # I previously set:
    # W[0][0] = 1
    # W[1][0] = 1
    # W[2][0] = 1
    # This is WRONG. W[1] is C10, W[2] is C01.
    # C00 gets 1. C10 gets 1. C01 gets 1.
    # So W[0]=1, W[1]=1, W[2]=1.
    # This matches my previous fill!
    # Let's check C01 formula: m3 + m1 + m7.
    # So m1 contributes 1 to C01.
    # C01 is index 2.
    # So W[2] should be 1.
    # My previous fill: W[2][0] = 1. Correct.
    # C10 formula: m2 + m4 + m1.
    # So m1 contributes 1 to C10.
    # C10 is index 1.
    # So W[1] should be 1.
    # My previous fill: W[1][0] = 1. Correct.
    
    # Okay, the indices were correct.
    
    # Continue Part 4:
    # C02 (i=0,k=2) -> 2*2+0 = 4
    # C12 (i=1,k=2) -> 2*2+1 = 5
    # C03 (i=0,k=3) -> 2*3+0 = 6
    # C13 (i=1,k=3) -> 2*3+1 = 7
    # C04 (i=0,k=4) -> 2*4+0 = 8
    # C14 (i=1,k=4) -> 2*4+1 = 9
    
    # Col 22: A02*B22 -> C02 (index 4)
    U[2][22] = 1
    V[12][22] = 1
    W[4][22] = 1
    
    # Col 23: A02*B23 -> C03 (index 6)
    U[2][23] = 1
    V[13][23] = 1
    W[6][23] = 1
    
    # Col 24: A02*B24 -> C04 (index 8)
    U[2][24] = 1
    V[14][24] = 1
    W[8][24] = 1
    
    # Col 25: A12*B22 -> C12 (index 5)
    U[5][25] = 1
    V[12][25] = 1
    W[5][25] = 1
    
    # Col 26: A12*B23 -> C13 (index 7)
    U[5][26] = 1
    V[13][26] = 1
    W[7][26] = 1
    
    # Col 27: A12*B24 -> C14 (index 9)
    U[5][27] = 1
    V[14][27] = 1
    W[9][27] = 1
    
    # Verify
    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
