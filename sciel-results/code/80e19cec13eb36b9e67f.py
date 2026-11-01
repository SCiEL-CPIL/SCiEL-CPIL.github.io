import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Rank = 49
    rank = 49
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Strassen coefficients
    # s is 0..6
    # U_coeffs[s][i, j]
    # V_coeffs[s][j, k']
    # W_coeffs[s][k', i]
    
    # A is 2x2. i, j in 0..1
    # B is 2x2 (block). j, k' in 0..1
    # C is 2x2 (block). i, k' in 0..1
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A10)B11
    # M6: (A11-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # U (A)
    U_c = [
        [[1, 0], [0, 1]], # s=0: A00, A11
        [[0, 0], [1, 1]], # s=1: A10, A11
        [[1, 0], [0, 0]], # s=2: A00
        [[0, 0], [0, 1]], # s=3: A11
        [[1, 1], [0, 0]], # s=4: A00, A10
        [[-1, 0], [0, 1]],# s=5: A00, A11
        [[0, 1], [0, -1]] # s=6: A01, A11
    ]
    
    # V (B)
    V_c = [
        [[1, 0], [0, 1]], # s=0: B00, B11
        [[1, 0], [0, 0]], # s=1: B00
        [[0, 1], [0, -1]],# s=2: B01, B11
        [[0, -1], [1, 0]],# s=3: B00, B10
        [[0, 0], [0, 1]], # s=4: B11
        [[1, 1], [0, 0]], # s=5: B00, B01
        [[0, 1], [1, 1]]  # s=6: B01, B11
    ]
    
    # W (C)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W[k', i]
    W_c = [
        [[1, 0], [0, 1]], # s=0: C00, C11
        [[0, 1], [-1, 0]],# s=1: C10, C11
        [[0, 1], [1, 0]], # s=2: C01, C11
        [[1, 0], [1, 0]], # s=3: C00, C10
        [[-1, 0], [1, 0]],# s=4: C00, C01
        [[0, 0], [1, 0]], # s=5: C11
        [[1, 0], [0, 0]]  # s=6: C00
    ]
    
    # Note: W_c indices are [k'][i]
    # s=0: C00 (k'=0, i=0) is 1. C11 (k'=1, i=1) is 1.
    # My list above:
    # s=0: [[1, 0], [0, 1]] -> W[0][0]=1, W[1][1]=1. Correct.
    # s=1: C10 (k'=0, i=1) is 1. C11 (k'=1, i=1) is -1.
    # List: [[0, 1], [-1, 0]] -> W[0][1]=1, W[1][1]=-1. Correct.
    # s=2: C01 (k'=1, i=0) is 1. C11 (k'=1, i=1) is 1.
    # List: [[0, 1], [1, 0]] -> W[0][1]=1? No.
    # C01 is i=0, k'=1. So W[1][0] should be 1.
    # My list W_c[2] is [[0, 1], [1, 0]].
    # W[0][0]=0, W[0][1]=1. W[1][0]=1, W[1][1]=0.
    # So W[1][0]=1 (C01) and W[0][1]=1 (C10)?
    # Wait. C01 is i=0, k'=1.
    # C10 is i=1, k'=0.
    # Formula: C01 = M3 + M5.
    # s=2 is M3.
    # So W[1][0] should be 1.
    # My list has W[1][0]=1. Correct.
    # Does M3 contribute to C10? No.
    # My list has W[0][1]=1. That is C10.
    # But M3 is $A_{00}(B_{01}-B_{11})$.
    # Does it contribute to C10?
    # $C_{10} = A_{10}B_{00} + A_{11}B_{10}$.
    # M3 involves $A_{00}$. So no contribution to C10.
    # So W[0][1] should be 0 for s=2.
    # Let's re-check W_c[2].
    # I wrote: `[[0, 1], [1, 0]]`.
    # This implies W[0][1]=1 and W[1][0]=1.
    # W[1][0] is C01 (i=0, k'=1). Correct.
    # W[0][1] is C10 (i=1, k'=0). Should be 0.
    # So W_c[2] should be `[[0, 0], [1, 1]]`?
    # C01 (i=0, k'=1) -> W[1][0]=1.
    # C11 (i=1, k'=1) -> W[1][1]=1.
    # So W_c[2] should be `[[0, 0], [1, 1]]`.
    
    # Let's re-verify all W_c.
    
    # s=0 (M1): C00 (1), C11 (1).
    # W[0][0]=1, W[1][1]=1.
    # W_c[0] = [[1, 0], [0, 1]]. Correct.
    
    # s=1 (M2): C10 (1), C11 (-1).
    # W[0][1]=1, W[1][1]=-1.
    # W_c[1] = [[0, 1], [0, -1]].
    # My previous: `[[0, 1], [-1, 0]]`.
    # W[1][0] is C01. Should be 0.
    # W[1][1] is C11. Should be -1.
    # So W_c[1] should be `[[0, 1], [0, -1]]`.
    
    # s=2 (M3): C01 (1), C11 (1).
    # W[1][0]=1, W[1][1]=1.
    # W_c[2] should be `[[0, 0], [1, 1]]`.
    
    # s=3 (M4): C00 (1), C10 (1).
    # W[0][0]=1, W[0][1]=1.
    # W_c[3] should be `[[1, 1], [0, 0]]`.
    
    # s=4 (M5): C00 (-1), C01 (1).
    # W[0][0]=-1, W[1][0]=1.
    # W_c[4] should be `[[-1, 0], [1, 0]]`.
    
    # s=5 (M6): C11 (1).
    # W[1][1]=1.
    # W_c[5] should be `[[0, 0], [0, 1]]`.
    
    # s=6 (M7): C00 (1).
    # W[0][0]=1.
    # W_c[6] should be `[[1, 0], [0, 0]]`.

    # Corrected W_c:
    W_c = [
        [[1, 0], [0, 1]], # s=0
        [[0, 1], [0, -1]],# s=1
        [[0, 0], [1, 1]], # s=2
        [[1, 1], [0, 0]], # s=3
        [[-1, 0], [1, 0]],# s=4
        [[0, 0], [0, 1]], # s=5
        [[1, 0], [0, 0]]  # s=6
    ]
    
    # Now populate
    num_blocks = p // 2 # 7
    
    for b in range(num_blocks):
        for s in range(7):
            r = b * 7 + s
            # U
            for i in range(2):
                for j in range(2):
                    U[i*2 + j, r] = U_c[s][i][j]
                    
            # V
            for j in range(2):
                for kp in range(2):
                    k = 2*b + kp
                    V[j*14 + k, r] = V_c[s][j][kp]
                    
            # W
            for kp in range(2):
                for i in range(2):
                    k = 2*b + kp
                    W[k*2 + i, r] = W_c[s][kp][i]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
