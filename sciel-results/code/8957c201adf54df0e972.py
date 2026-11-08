import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    # Part 1: A_R B_R (2x1 by 1x10) -> 20 terms
    # Part 2: A_L B_L (2x2 by 2x10) -> 5 blocks of 2x2, each Strassen (7 terms) -> 35 terms
    total_rank = 20 + 35
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_R B_R
    # A_R is column j=2 of A. B_R is row j=2 of B.
    # C[i, k] += A[i, 2] * B[2, k]
    for k in range(p):
        for i in range(n):
            # U index: i * m + 2
            U[i * m + 2][col] = 1
            # V index: 2 * p + k
            V[2 * p + k][col] = 1
            # W index: k * n + i
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A_L B_L
    # A_L is columns j=0,1. B_L is rows j=0,1.
    # Split B_L into 5 blocks of 2 columns.
    for t in range(5):
        k1 = 2 * t
        k2 = 2 * t + 1
        
        # Strassen variables for 2x2 multiplication
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # m1 = (A00 + A11) * (B00 + B11)
        # m2 = (A10 + A11) * B00
        # m3 = A00 * (B01 - B11)
        # m4 = A11 * (B10 - B00)
        # m5 = (A00 + A01) * B11
        # m6 = (A10 - A11) * (B00 + B01)
        # m7 = (A01 - A11) * (B10 + B11)
        
        # We need to map these to U, V, W.
        # U indices for A_L:
        # A00: i=0, j=0 -> idx 0
        # A01: i=0, j=1 -> idx 1
        # A10: i=1, j=0 -> idx 3
        # A11: i=1, j=1 -> idx 4
        
        # V indices for B_L block t:
        # B00: j=0, k=k1 -> idx 0*p + k1 = k1
        # B01: j=0, k=k2 -> idx 0*p + k2 = k2
        # B10: j=1, k=k1 -> idx 1*p + k1 = p + k1
        # B11: j=1, k=k2 -> idx 1*p + k2 = p + k2
        
        # W indices for C_L block t:
        # C00: i=0, k=k1 -> idx k1*n + 0 = 2*k1
        # C01: i=0, k=k2 -> idx k2*n + 0 = 2*k2
        # C10: i=1, k=k1 -> idx k1*n + 1 = 2*k1 + 1
        # C11: i=1, k=k2 -> idx k2*n + 1 = 2*k2 + 1
        
        # Define the 7 Strassen terms
        # Each term is (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs is dict {idx: val}, V_coeffs is dict {idx: val}, W_coeffs is dict {idx: val}
        
        # m1 = (A00 + A11) * (B00 + B11)
        U1 = {0: 1, 4: 1}
        V1 = {k1: 1, p + k2: 1} # Wait, B00 is j=0,k1. B11 is j=1,k2.
        # B00 idx: 0*p + k1 = k1
        # B11 idx: 1*p + k2 = p + k2
        # Correct.
        W1 = {2*k1: 1, 2*k2: 1, 2*k1+1: -1, 2*k2+1: -1}
        # C00 += m1, C01 += m1, C10 -= m1, C11 -= m1?
        # Strassen:
        # C00 = m1 + m4 - m5 - m7
        # C01 = m2 + m4
        # C10 = m3 + m5
        # C11 = m1 - m2 + m3 + m6
        
        # So for m1:
        # W[2*k1] += 1
        # W[2*k2] += 1
        # W[2*k1+1] -= 1
        # W[2*k2+1] -= 1
        # Wait, C10 is i=1, k=k1. C11 is i=1, k=k2.
        # C10 coeff for m1 is -1.
        # C11 coeff for m1 is -1? No, C11 = m1 - m2 + m3 + m6. So coeff is 1.
        # Let's re-verify Strassen.
        # C = [[m1+m4-m5-m7, m2+m4], [m3+m5, m1-m2+m3+m6]]
        # So:
        # C00: m1:1, m4:1, m5:-1, m7:-1
        # C01: m2:1, m4:1
        # C10: m3:1, m5:1
        # C11: m1:1, m2:-1, m3:1, m6:1
        
        # So for m1:
        W1 = {2*k1: 1, 2*k2: 1, 2*k1+1: 1, 2*k2+1: 1} # C11 has m1 with +1. C10 does NOT have m1.
        # Wait, C10 = m3+m5. No m1.
        # So W1[2*k1+1] should be 0.
        # W1[2*k2+1] should be 1.
        
        # Let's list the contributions carefully.
        
        # Term 1: m1
        U1 = {0: 1, 4: 1}
        V1 = {k1: 1, p + k2: 1}
        W1 = {2*k1: 1, 2*k2: 1, 2*k2+1: 1} # C00, C01, C11
        
        # Term 2: m2 = (A10 + A11) * B00
        U2 = {3: 1, 4: 1}
        V2 = {k1: 1}
        W2 = {2*k2: 1, 2*k2+1: -1} # C01: +1, C11: -1
        
        # Term 3: m3 = A00 * (B01 - B11)
        U3 = {0: 1}
        V3 = {k2: 1, p + k2: -1}
        W3 = {2*k1+1: 1, 2*k2+1: 1} # C10: +1, C11: +1
        
        # Term 4: m4 = A11 * (B10 - B00)
        U4 = {4: 1}
        V4 = {p + k1: 1, k1: -1}
        W4 = {2*k1: 1, 2*k2: 1} # C00: +1, C01: +1
        
        # Term 5: m5 = (A00 + A01) * B11
        U5 = {0: 1, 1: 1}
        V5 = {p + k2: 1}
        W5 = {2*k1: -1, 2*k1+1: 1} # C00: -1, C10: +1
        
        # Term 6: m6 = (A10 - A11) * (B00 + B01)
        U6 = {3: 1, 4: -1}
        V6 = {k1: 1, k2: 1}
        W6 = {2*k2+1: 1} # C11: +1
        
        # Term 7: m7 = (A01 - A11) * (B10 + B11)
        U7 = {1: 1, 4: -1}
        V7 = {p + k1: 1, p + k2: 1}
        W7 = {2*k1: -1} # C00: -1
        
        # Now add these to U, V, W
        terms = [
            (U1, V1, W1),
            (U2, V2, W2),
            (U3, V3, W3),
            (U4, V4, W4),
            (U5, V5, W5),
            (U6, V6, W6),
            (U7, V7, W7)
        ]
        
        for u_dict, v_dict, w_dict in terms:
            for idx, val in u_dict.items():
                U[idx][col] += val
            for idx, val in v_dict.items():
                V[idx][col] += val
            for idx, val in w_dict.items():
                W[idx][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
