import numpy as np

def construct():
    n, m, p = 2, 3, 6
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A_right (col 2) * B_bottom (row 2)
    # This is a rank 1 term: A[i,2] * B[2,k]
    r = 0
    for i in range(n):
        U[i * m + 2][r] = 1
    for k in range(p):
        V[2 * p + k][r] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][r] = 1
    
    # Part 2: A_left (cols 0,1) * B_top (rows 0,1)
    # Split B_top into 3 blocks of width 2.
    # Use Strassen's algorithm for each 2x2 block.
    
    # Strassen coefficients for A (indices 00, 01, 10, 11)
    # A_global indices:
    # 00 -> U[0*3+0] = U[0]
    # 01 -> U[0*3+1] = U[1]
    # 10 -> U[1*3+0] = U[3]
    # 11 -> U[1*3+1] = U[4]
    
    # Strassen terms:
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Coeffs for U (A)
    U_coeffs = [
        {0: 1, 4: 1},          # M1: A00, A11
        {3: 1, 4: 1},          # M2: A10, A11
        {0: 1},                # M3: A00
        {4: 1},                # M4: A11
        {0: 1, 1: 1},          # M5: A00, A01
        {3: 1, 0: -1},         # M6: A10, A00
        {1: 1, 4: -1}          # M7: A01, A11
    ]
    
    # Coeffs for V (B) - relative to block start
    # Local B indices: 00, 01, 10, 11
    # Global V index = j*6 + k
    # For block b:
    # 00 -> 0*6 + 2b
    # 01 -> 0*6 + 2b+1
    # 10 -> 1*6 + 2b
    # 11 -> 1*6 + 2b+1
    
    V_coeffs = [
        {0: 1, 1: 1},          # M1: B00, B11 (local)
        {0: 1},                # M2: B00
        {1: 1, 3: -1},         # M3: B01, B11
        {2: 1, 0: -1},         # M4: B10, B00
        {3: 1},                # M5: B11
        {0: 1, 1: 1},          # M6: B00, B01
        {2: 1, 3: 1}           # M7: B10, B11
    ]
    
    # Coeffs for W (C)
    # Local C indices: 00, 01, 10, 11
    # Global W index = k*2 + i
    # For block b:
    # 00 -> 2b*2 + 0 = 4b
    # 01 -> (2b+1)*2 + 0 = 4b+2
    # 10 -> 2b*2 + 1 = 4b+1
    # 11 -> (2b+1)*2 + 1 = 4b+3
    
    W_coeffs = [
        {0: 1, 3: 1},          # M1: C00, C11
        {2: 1, 3: -1},         # M2: C10, C11
        {1: 1, 3: 1},          # M3: C01, C11
        {0: 1, 2: 1},          # M4: C00, C10
        {0: -1, 1: 1},         # M5: C00, C01
        {3: 1},                # M6: C11
        {0: 1}                 # M7: C00
    ]
    
    r = 1
    for b in range(3):
        for t in range(7):
            # Set U
            for u_idx, val in U_coeffs[t].items():
                U[u_idx][r] = val
            
            # Set V
            for v_local, val in V_coeffs[t].items():
                # v_local 0->00, 1->01, 2->10, 3->11
                if v_local == 0:
                    v_idx = 0 * p + (2 * b)
                elif v_local == 1:
                    v_idx = 0 * p + (2 * b + 1)
                elif v_local == 2:
                    v_idx = 1 * p + (2 * b)
                elif v_local == 3:
                    v_idx = 1 * p + (2 * b + 1)
                V[v_idx][r] = val
            
            # Set W
            for w_local, val in W_coeffs[t].items():
                # w_local 0->00, 1->01, 2->10, 3->11
                if w_local == 0:
                    w_idx = (2 * b) * n + 0
                elif w_local == 1:
                    w_idx = (2 * b + 1) * n + 0
                elif w_local == 2:
                    w_idx = (2 * b) * n + 1
                elif w_local == 3:
                    w_idx = (2 * b + 1) * n + 1
                W[w_idx][r] = val
            
            r += 1
            
    return {"U": U, "V": V, "W": W}
