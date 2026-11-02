import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = 4, 5, 6  # A is n×m, B is m×p, C = A*B is n×p
    # C[k*n+i, r] convention: W index is k*n+i
    
    # Partition: i: 4=2+2, j: 5=2+2+1, k: 6=2+2+2
    i_blocks = [(0, 2), (2, 4)]  # size 2 each
    j_blocks = [(0, 2), (2, 4), (4, 5)]  # sizes 2, 2, 1
    k_blocks = [(0, 2), (2, 4), (4, 6)]  # size 2 each
    
    rank = 108  # 12 * 7 + 6 * 4
    U = [[0] * rank for _ in range(n * m)]  # 20 x 108
    V = [[0] * rank for _ in range(m * p)]  # 30 x 108
    W = [[0] * rank for _ in range(p * n)]  # 24 x 108
    
    col = 0
    
    # Strassen coefficients for 2x2x2:
    # Each term: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: coefficients for [A00, A01, A10, A11]
    # b_coeffs: coefficients for [B00, B01, B10, B11]
    # c_coeffs: coefficients for [C00, C01, C10, C11]
    strassen_terms = [
        # M1 = (A00+A11)*(B00+B11) -> C00 += 1, C11 -= 1
        ([1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, -1]),  # wait, this isn't right
    ]
    # Let me redo Strassen properly.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each term t, the contribution to C is:
    # term t contributes c_t[i] * M_t to C[i]
    # where M_t = (sum a_t[j]*A[j]) * (sum b_t[k]*B[k])
    
    # So the rank-1 decomposition: for each term t,
    # U[:, col] = a_t (padded to full A size)
    # V[:, col] = b_t (padded to full B size)
    # W[:, col] = c_t (padded to full C size)
    
    strassen = [
        # (a_coeffs for [A00,A01,A10,A11], b_coeffs for [B00,B01,B10,B11], c_coeffs for [C00,C01,C10,C11])
        ([1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, -1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, -1, 0]),   # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),   # M7
    ]
    
    # Verify Strassen:
    # C00 = M1+M4-M5+M7: c[0] = 1+1-(-1)+1 = 4? That's wrong.
    # Let me recheck.
    # C00 = M1+M4-M5+M7
    #   M1 contributes +1 to C00
    #   M4 contributes +1 to C00
    #   M5 contributes -1 to C00
    #   M7 contributes +1 to C00
    # So c_coeffs for C00: M1:1, M4:1, M5:-1, M7:1
    # But in my list above, I have the c_coeffs per term:
    # Term 0 (M1): c = [1,0,0,-1] means M1 contributes +1 to C00, 0 to C01, 0 to C10, -1 to C11
    # Term 1 (M2): c = [0,0,1,-1] means M2 contributes 0 to C00, 0 to C01, +1 to C10, -1 to C11
    # Term 2 (M3): c = [0,1,0,1] means M3 contributes 0 to C00, +1 to C01, 0 to C10, +1 to C11
    # Term 3 (M4): c = [1,0,1,0] means M4 contributes +1 to C00, 0 to C01, +1 to C10, 0 to C11
    # Term 4 (M5): c = [0,1,-1,0] means M5 contributes 0 to C00, +1 to C01, -1 to C10, 0 to C11
    # Term 5 (M6): c = [0,0,0,1] means M6 contributes 0 to C00, 0 to C01, 0 to C10, +1 to C11
    # Term 6 (M7): c = [1,0,0,1] means M7 contributes +1 to C00, 0 to C01, 0 to C10, +1 to C11
    
    # Check C00: M1(1) + M4(1) + M5(-1) + M7(1) = 1+1-1+1 = 2? But should be 1.
    # Hmm, I think I have an error. Let me recompute.
    
    # Actually, the standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # So the coefficient of M_t in C00 is:
    # M1: +1, M2: 0, M3: 0, M4: +1, M5: -1, M6: 0, M7: +1
    # C01 = M3 + M5: M1:0, M2:0, M3:+1, M4:0, M5:+1, M6:0, M7:0
    # C10 = M2 + M4: M1:0, M2:+1, M3:0, M4:+1, M5:0, M6:0, M7:0
    # C11 = M1 - M2 + M3 + M6: M1:+1, M2:-1, M3:+1, M4:0, M5:0, M6:+1, M7:0
    
    # So the c_coeffs per term (as column vectors [C00,C01,C10,C11]):
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, -1]
    # M3: [0, 1, 0, 1]
    # M4: [1, 0, 1, 0]
    # M5: [-1, 1, 0, 0]
    # M6: [0, 0, 0, 1]
    # M7: [1, 0, 0, 0]
    
    # Let me verify C00 = 1+0+0+1-1+0+1 = 2. That's still 2, not 1.
    # Wait, I think I'm confusing myself. Let me just verify with actual Strassen.
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Let me verify C00 with A = [[1,0],[0,1]], B = [[1,0],[0,1]] (identity*identity = identity)
    # M1 = (1+1)(1+1) = 4
    # M2 = (0+1)(1) = 1
    # M3 = 1(0-1) = -1
    # M4 = 1(0-1) = -1
    # M5 = (1+0)(1) = 1
    # M6 = (0-1)(1+0) = -1
    # M7 = (0-1)(0+1) = -1
    # C00 = 4+(-1)-1+(-1) = 1 ✓
    # C01 = -1+1 = 0 ✓
    # C10 = 1+(-1) = 0 ✓
    # C11 = 4-1+(-1)+(-1) = 1 ✓
    
    # Good. Now the c_coeffs:
    # C00 = 1*M1 + 0*M2 + 0*M3 + 1*M4 + (-1)*M5 + 0*M6 + 1*M7
    # C01 = 0*M1 + 0*M2 + 1*M3 + 0*M4 + 1*M5 + 0*M6 + 0*M7
    # C10 = 0*M1 + 1*M2 + 0*M3 + 1*M4 + 0*M5 + 0*M6 + 0*M7
    # C11 = 1*M1 + (-1)*M2 + 1*M3 + 0*M4 + 0*M5 + 1*M6 + 0*M7
    
    # So per term, the c vector [C00,C01,C10,C11]:
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, -1]
    # M3: [0, 1, 0, 1]
    # M4: [1, 0, 1, 0]
    # M5: [-1, 1, 0, 0]
    # M6: [0, 0, 0, 1]
    # M7: [1, 0, 0, 0]
    
    strassen = [
        # (a_coeffs [A00,A01,A10,A11], b_coeffs [B00,B01,B10,B11], c_coeffs [C00,C01,C10,C11])
        ([1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
    ]
    
    col = 0
    for i_idx, (i0, i1) in enumerate(i_blocks):
        for j_idx, (j0, j1) in enumerate(j_blocks):
            for k_idx, (k0, k1) in enumerate(k_blocks):
                j_size = j1 - j0
                if j_size == 2:
                    # (2, 2, 2) block - use Strassen
                    for t in range(7):
                        a_c, b_c, c_c = strassen[t]
                        # U: A indices i0..i1, j0..j1
                        for di in range(2):
                            for dj in range(2):
                                U[(i0+di)*m + (j0+dj)][col] += a_c[di*2+dj]
                        # V: B indices j0..j1, k0..k1
                        for dj in range(2):
                            for dk in range(2):
                                V[(j0+dj)*p + (k0+dk)][col] += b_c[dj*2+dk]
                        # W: C indices k0..k1, i0..i1 (W[k*n+i])
                        for dk in range(2):
                            for di in range(2):
                                W[(k0+dk)*n + (i0+di)][col] += c_c[dk*2+di]
                        col += 1
                else:
                    # (2, 1, 2) block - schoolbook, 4 terms
                    # A is 2x1 at [i0:i0+2, j0:j0+1], B is 1x2 at [j0:j0+1, k0:k0+2]
                    # C is 2x2 at [k0:k0+2, i0:i0+2]
                    # C[di][dk] = A[di][0] * B[0][dk]
                    for di in range(2):
                        for dk in range(2):
                            U[(i0+di)*m + j0][col] = 1
                            V[j0*p + (k0+dk)][col] = 1
                            W[(k0+dk)*n + (i0+di)][col] = 1
                            col += 1
    
    return {"U": U, "V": V, "W": W}
