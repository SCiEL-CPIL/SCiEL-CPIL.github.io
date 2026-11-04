import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 40 decomposition
    # Split B (3x7) into B1 (3x3) and B2 (3x4)
    # B2 split into B21 (3x2) and B22 (3x2)
    
    # Part 1: 2x3 * 3x3 (Schoolbook, rank 18)
    # Part 2: 2x3 * 3x2 (Rank 11)
    # Part 3: 2x3 * 3x2 (Rank 11)
    
    rank = 40
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: 2x3 * 3x3
    # Indices: A (2x3), B (3x3), C (2x3)
    # B columns 0,1,2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                a_idx = i * m + j
                b_idx = j * 3 + k # B1 is 3x3, so j in 0..2, k in 0..2
                c_idx = k * n + i
                U[a_idx, col] = 1.0
                V[b_idx, col] = 1.0
                W[c_idx, col] = 1.0
                col += 1
                
    # Part 2: 2x3 * 3x2 (B21, columns 3,4)
    # Use Rank 11 algorithm for 2x3x2
    # This is 2x2x2 (Strassen) + 2x2x1 (Schoolbook)
    # A is 2x3. B21 is 3x2.
    # Split A into A1 (2x2) and A2 (2x1).
    # Split B21 into B11 (2x2) and B12 (1x2).
    # C = A1 B11 + A2 B12
    
    # A1 (2x2) part: Rows 0,1 of A. Cols 0,1 of B21 (which are global cols 3,4? No, B21 is 3x2).
    # B21 indices: j in 0..2, k in 0..1 (local). Global k is 3+k_local.
    # B11 is top-left 2x2 of B21. j in 0..1, k_local in 0..1.
    # B12 is bottom row of B21. j=2, k_local in 0..1.
    
    # Strassen for A1 (2x2) * B11 (2x2)
    # A1 indices: i in 0..1, j in 0..1.
    # B11 indices: j in 0..1, k_local in 0..1.
    # C11 indices: i in 0..1, k_local in 0..1.
    
    # Strassen factors for 2x2x2
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # We need to map these to the global U, V, W.
    # U rows: A indices.
    # V rows: B indices.
    # W rows: C indices.
    
    # Strassen 1: (A00+A11)(B00+B11)
    # A00: i=0, j=0. A11: i=1, j=1.
    # B00: j=0, k_local=0. B11: j=1, k_local=1.
    # C00: i=0, k_local=0. C11: i=1, k_local=1.
    # C01: i=0, k_local=1. C10: i=1, k_local=0.
    
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # This contributes to C00, C01, C10, C11.
    # C00 += A00B00
    # C01 += A00B11
    # C10 += A11B00
    # C11 += A11B11
    
    # In the tensor decomposition, each M is a rank-1 term.
    # M1 = (u1 . A) (v1 . B) -> adds to w1 . C
    # u1 = e_A00 + e_A11
    # v1 = e_B00 + e_B11
    # w1 = e_C00 + e_C11 ? No.
    # The product (A00+A11)(B00+B11) gives a scalar.
    # This scalar is added to C00 and C11?
    # C00 = A00B00 + A01B10 + ...
    # C11 = A10B01 + A11B11 + ...
    # Strassen:
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    
    # So for M1, the W vector has 1 at C00 and 1 at C11.
    
    # Let's implement the 7 Strassen terms for the 2x2x2 part.
    # Then 4 Schoolbook terms for the 2x2x1 part.
    
    # Strassen Terms for Part 2 (A1 * B11)
    # A1 is A[:, 0:2]
    # B11 is B[0:2, 3:5] (Global indices)
    # C11 is C[:, 3:5]
    
    def add_strassen_term(col, u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: dict {a_idx: val}
        # v_coeffs: dict {b_idx: val}
        # w_coeffs: dict {c_idx: val}
        for idx, val in u_coeffs.items():
            U[idx, col] = val
        for idx, val in v_coeffs.items():
            V[idx, col] = val
        for idx, val in w_coeffs.items():
            W[idx, col] = val
            # Note: W index is c_idx = k*n + i
            # For C11 (i, k_local), global k = 3 + k_local
            # c_idx = (3+k_local)*n + i

    # Strassen 1: (A00+A11)(B00+B11)
    # A00: i=0, j=0 -> 0
    # A11: i=1, j=1 -> 1*3+1 = 4
    # B00: j=0, k_local=0 -> 0*3 + 3 = 3 (Global B index: j*p + k_global. j=0, k_global=3)
    # B11: j=1, k_local=1 -> 1*3 + 4 = 7 (j=1, k_global=4)
    # C00: i=0, k_local=0 -> 3*2 + 0 = 6 (Global C index: k_global*n + i. k_global=3, i=0)
    # C11: i=1, k_local=1 -> 4*2 + 1 = 9 (k_global=4, i=1)
    
    u1 = {0: 1, 4: 1}
    v1 = {3: 1, 7: 1}
    w1 = {6: 1, 9: 1}
    add_strassen_term(col, u1, v1, w1)
    col += 1

    # Strassen 2: (A10+A11)B00
    # A10: i=1, j=0 -> 3
    # A11: i=1, j=1 -> 4
    # B00: j=0, k_local=0 -> 3
    # C10: i=1, k_local=0 -> 3*2+1 = 7
    u2 = {3: 1, 4: 1}
    v2 = {3: 1}
    w2 = {7: 1}
    add_strassen_term(col, u2, v2, w2)
    col += 1

    # Strassen 3: A00(B10-B11)
    # A00: 0
    # B10: j=1, k_local=0 -> 1*3+3 = 6
    # B11: j=1, k_local=1 -> 7
    # C01: i=0, k_local=1 -> 4*2+0 = 8
    u3 = {0: 1}
    v3 = {6: 1, 7: -1}
    w3 = {8: 1}
    add_strassen_term(col, u3, v3, w3)
    col += 1

    # Strassen 4: A11(B01-B00)
    # A11: 4
    # B01: j=0, k_local=1 -> 0*3+4 = 4
    # B00: 3
    # C10: 7
    u4 = {4: 1}
    v4 = {4: 1, 3: -1}
    w4 = {7: 1}
    add_strassen_term(col, u4, v4, w4)
    col += 1

    # Strassen 5: (A00+A01)B11
    # A00: 0
    # A01: i=0, j=1 -> 1
    # B11: 7
    # C01: 8
    u5 = {0: 1, 1: 1}
    v5 = {7: 1}
    w5 = {8: 1}
    add_strassen_term(col, u5, v5, w5)
    col += 1

    # Strassen 6: (A10-A00)(B00+B01)
    # A10: 3
    # A00: 0
    # B00: 3
    # B01: 4
    # C10: 7
    u6 = {3: 1, 0: -1}
    v6 = {3: 1, 4: 1}
    w6 = {7: 1}
    add_strassen_term(col, u6, v6, w6)
    col += 1

    # Strassen 7: (A01-A11)(B10+B11)
    # A01: 1
    # A11: 4
    # B10: 6
    # B11: 7
    # C11: 9
    u7 = {1: 1, 4: -1}
    v7 = {6: 1, 7: 1}
    w7 = {9: 1}
    add_strassen_term(col, u7, v7, w7)
    col += 1

    # Schoolbook for A2 (2x1) * B12 (1x2)
    # A2 is A[:, 2] (j=2)
    # B12 is B[2, 3:5] (j=2, k_global=3,4)
    # C12 is C[:, 3:5]
    
    # Terms:
    # C03 += A02 B23
    # C04 += A02 B24
    # C13 += A12 B23
    # C14 += A12 B24
    
    # A02: i=0, j=2 -> 2
    # A12: i=1, j=2 -> 5
    # B23: j=2, k_global=3 -> 2*3+3 = 9
    # B24: j=2, k_global=4 -> 2*3+4 = 10
    # C03: i=0, k_global=3 -> 3*2+0 = 6
    # C04: i=0, k_global=4 -> 4*2+0 = 8
    # C13: i=1, k_global=3 -> 3*2+1 = 7
    # C14: i=1, k_global=4 -> 4*2+1 = 9
    
    # Term 8: A02 B23 -> C03
    U[2, col] = 1
    V[9, col] = 1
    W[6, col] = 1
    col += 1
    
    # Term 9: A02 B24 -> C04
    U[2, col] = 1
    V[10, col] = 1
    W[8, col] = 1
    col += 1
    
    # Term 10: A12 B23 -> C13
    U[5, col] = 1
    V[9, col] = 1
    W[7, col] = 1
    col += 1
    
    # Term 11: A12 B24 -> C14
    U[5, col] = 1
    V[10, col] = 1
    W[9, col] = 1
    col += 1

    # Part 3: 2x3 * 3x2 (B22, columns 5,6)
    # Similar to Part 2, but global k is 5,6.
    # B22 is 3x2. j in 0..2, k_local in 0..1. Global k = 5+k_local.
    
    # Strassen for A1 (2x2) * B21_part (2x2)
    # B21_part is B[0:2, 5:7].
    # B00_local: j=0, k_global=5 -> 0*3+5 = 5
    # B01_local: j=0, k_global=6 -> 0*3+6 = 6
    # B10_local: j=1, k_global=5 -> 1*3+5 = 8
    # B11_local: j=1, k_global=6 -> 1*3+6 = 9
    
    # C indices:
    # C00_local: i=0, k_global=5 -> 5*2+0 = 10
    # C01_local: i=0, k_global=6 -> 6*2+0 = 12
    # C10_local: i=1, k_global=5 -> 5*2+1 = 11
    # C11_local: i=1, k_global=6 -> 6*2+1 = 13
    
    # Term 12: (A00+A11)(B00+B11)
    u12 = {0: 1, 4: 1}
    v12 = {5: 1, 9: 1}
    w12 = {10: 1, 13: 1}
    add_strassen_term(col, u12, v12, w12)
    col += 1

    # Term 13: (A10+A11)B00
    u13 = {3: 1, 4: 1}
    v13 = {5: 1}
    w13 = {11: 1}
    add_strassen_term(col, u13, v13, w13)
    col += 1

    # Term 14: A00(B10-B11)
    u14 = {0: 1}
    v14 = {8: 1, 9: -1}
    w14 = {12: 1}
    add_strassen_term(col, u14, v14, w14)
    col += 1

    # Term 15: A11(B01-B00)
    u15 = {4: 1}
    v15 = {6: 1, 5: -1}
    w15 = {11: 1}
    add_strassen_term(col, u15, v15, w15)
    col += 1

    # Term 16: (A00+A01)B11
    u16 = {0: 1, 1: 1}
    v16 = {9: 1}
    w16 = {12: 1}
    add_strassen_term(col, u16, v16, w16)
    col += 1

    # Term 17: (A10-A00)(B00+B01)
    u17 = {3: 1, 0: -1}
    v17 = {5: 1, 6: 1}
    w17 = {11: 1}
    add_strassen_term(col, u17, v17, w17)
    col += 1

    # Term 18: (A01-A11)(B10+B11)
    u18 = {1: 1, 4: -1}
    v18 = {8: 1, 9: 1}
    w18 = {13: 1}
    add_strassen_term(col, u18, v18, w18)
    col += 1

    # Schoolbook for A2 (2x1) * B22_part (1x2)
    # B22_part is B[2, 5:7].
    # B25: j=2, k_global=5 -> 2*3+5 = 11
    # B26: j=2, k_global=6 -> 2*3+6 = 12
    
    # C05: i=0, k_global=5 -> 10
    # C06: i=0, k_global=6 -> 12
    # C15: i=1, k_global=5 -> 11
    # C16: i=1, k_global=6 -> 13
    
    # Term 19: A02 B25 -> C05
    U[2, col] = 1
    V[11, col] = 1
    W[10, col] = 1
    col += 1
    
    # Term 20: A02 B26 -> C06
    U[2, col] = 1
    V[12, col] = 1
    W[12, col] = 1
    col += 1
    
    # Term 21: A12 B25 -> C15
    U[5, col] = 1
    V[11, col] = 1
    W[11, col] = 1
    col += 1
    
    # Term 22: A12 B26 -> C16
    U[5, col] = 1
    V[12, col] = 1
    W[13, col] = 1
    col += 1

    # Total rank used: 18 + 11 + 11 = 40.
    # Check col: 18 + 7 + 4 + 7 + 4 = 40. Correct.

    U_final = [[int(val) for val in row] for row in U.tolist()]
    V_final = [[int(val) for val in row] for row in V.tolist()]
    W_final = [[int(val) for val in row] for row in W.tolist()]
    
    return {"U": U_final, "V": V_final, "W": W_final}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
