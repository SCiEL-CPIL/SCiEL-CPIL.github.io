import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=0, m=1, p=5
    # n=2, m=1, p=5
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j in {1, 2}
    # Map local j' in {0, 1} to global j = j' + 1
    
    # Sub-part 2a: k in {0, 1} (Strassen 2x2x2)
    # Local k' in {0, 1} maps to global k = k'
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: i, j' ; V: j', k' ; W: k', i
        # P1: (0,0)+(1,1) for U, V, W
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # P2: (1,0)+(1,1) for U, (0,0) for V, (0,0) for W? No.
        # P2 = (A10+A11)B00
        # U: (1,0):1, (1,1):1. V: (0,0):1. W: (0,0):1.
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 1}),
        # P3: A00(B01-B11)
        # U: (0,0):1. V: (0,1):1, (1,1):-1. W: (1,0):1? No.
        # P3 contributes to C01.
        # C01 = P3 + P5.
        # P3 = A00(B01-B11).
        # Term: A00 * B01 -> C01. A00 * B11 -> -C01.
        # U: (0,0):1.
        # V: (0,1):1, (1,1):-1.
        # W: (1,0):1. (Since C01 is i=0, k=1. W index is k*n+i = 1*2+0 = 2. W[2]=1)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1}),
        # P4: A11(B10-B00)
        # Contributes to C10 and C11?
        # C10 = P2 + P4. C11 = P1 - P2 + P3 + P6.
        # P4 = A11(B10-B00).
        # A11*B10 -> C11? No, A11*B10 contributes to C10?
        # C10 = sum_j A1j Bj0 = A10 B00 + A11 B10.
        # So A11 B10 -> C10.
        # A11 B00 -> -C10? No.
        # P4 is used in C10 and C11.
        # C10 += P4. C11 += P4?
        # Let's check Strassen formula again.
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        # P4 = A11(B10 - B00).
        # In C10: A11 B10 is part of C10. A11 B00 is not.
        # Wait, C10 = A10 B00 + A11 B10.
        # P2 = (A10+A11)B00 = A10 B00 + A11 B00.
        # P4 = A11 B10 - A11 B00.
        # P2 + P4 = A10 B00 + A11 B10. Correct.
        # So P4 contributes to C10.
        # Does P4 contribute to C11?
        # C11 = P1 - P2 + P3 + P6. P4 is not there.
        # So P4 only contributes to C10?
        # Wait, P4 is a term. It must have a W vector.
        # If it only contributes to C10, then W should be non-zero only at (1,0).
        # U: (1,1):1.
        # V: (1,0):1, (0,0):-1.
        # W: (0,1):1? No, C10 is i=1, k=0. W index 0*2+1 = 1. W[1]=1.
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1}),
        # P5: (A00+A01)B11
        # Contributes to C01 and C00.
        # C01 += P5. C00 -= P5.
        # P5 = A00 B11 + A01 B11.
        # A01 B11 -> C01.
        # A00 B11 -> -C00.
        # U: (0,0):1, (0,1):1.
        # V: (1,1):1.
        # W: (1,1):1 for C01? No.
        # W must handle both C00 and C01?
        # No, a rank-1 tensor has a single W vector.
        # If W has non-zero entries at (0,0) and (1,0) (for C00 and C01?),
        # W index for C00 (i=0, k=0) is 0.
        # W index for C01 (i=0, k=1) is 2.
        # If W[0] = -1 and W[2] = 1.
        # Then term is -1 * (A00+A01)B11 for C00 and 1 * (A00+A01)B11 for C01.
        # Correct.
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        # P6: (A10-A00)(B00+B01)
        # Contributes to C11.
        # C11 += P6.
        # P6 = A10 B00 + A10 B01 - A00 B00 - A00 B01.
        # U: (1,0):1, (0,0):-1.
        # V: (0,0):1, (0,1):1.
        # W: (1,1):1. (C11 is i=1, k=1. W index 1*2+1=3).
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # P7: (A01-A11)(B10+B11)
        # Contributes to C00.
        # C00 += P7.
        # P7 = A01 B10 + A01 B11 - A11 B10 - A11 B11.
        # U: (0,1):1, (1,1):-1.
        # V: (1,0):1, (1,1):1.
        # W: (0,0):1. (C00 is i=0, k=0. W index 0).
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for u_dict, v_dict, w_dict in strassen_terms:
        for (i, jp), val_u in u_dict.items():
            U[i * m + (jp + 1)][col] += val_u
        for (jp, kp), val_v in v_dict.items():
            V[(jp + 1) * p + kp][col] += val_v
        for (kp, i), val_w in w_dict.items():
            W[kp * n + i][col] += val_w
        col += 1
        
    # Sub-part 2b: k in {2, 3} (Strassen 2x2x2)
    # Local k' in {0, 1} maps to global k = k' + 2
    for u_dict, v_dict, w_dict in strassen_terms:
        for (i, jp), val_u in u_dict.items():
            U[i * m + (jp + 1)][col] += val_u
        for (jp, kp), val_v in v_dict.items():
            V[(jp + 1) * p + (kp + 2)][col] += val_v
        for (kp, i), val_w in w_dict.items():
            W[(kp + 2) * n + i][col] += val_w
        col += 1
        
    # Sub-part 2c: k in {4} (Schoolbook 2x2x1)
    schoolbook_221 = [
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),
        ({(1,0): 1}, {(0,0): 1}, {(1,0): 1}),
        ({(1,1): 1}, {(1,0): 1}, {(1,0): 1}),
    ]
    
    for u_dict, v_dict, w_dict in schoolbook_221:
        for (i, jp), val_u in u_dict.items():
            U[i * m + (jp + 1)][col] += val_u
        for (jp, kp), val_v in v_dict.items():
            V[(jp + 1) * p + 4][col] += val_v
        for (kp, i), val_w in w_dict.items():
            W[4 * n + i][col] += val_w
        col += 1
        
    return {"U": U, "V": V, "W": W}
